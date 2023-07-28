import random
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Optional, List
import gurobipy as gp
import warnings
import networkx as nx
import matplotlib.pyplot as plt


class Instance:
    def __init__(self, N_size, K_size, T_max, C_size, t=None, alpha=None, seed=None, full_name=None,
                 optimal_value=None):
        self.name = None if full_name is None else full_name[3:-4]
        self.id = None if full_name is None else full_name[:2]
        self.optimal_value = optimal_value

        if seed is not None:
            random.seed(seed)

        self.N = {i + 1 for i in range(N_size)}
        self.K = {i + 1 for i in range(K_size)}
        self.N_0 = {0}.union(self.N)
        self.t = {
            (i, j): 0 if i == j else random.randint(1, 10)
            for i in self.N_0
            for j in self.N_0
        } if t is None else t
        self.T_max = T_max
        self.C = {i + 1 for i in range(C_size)}
        self.alpha = {
            (i, c): 1 if random.random() < 0.25 else 0
            for c in self.C
            for i in self.N
        } if alpha is None else alpha
        self.tau = {
            c: sum(self.alpha[i, c] for i in self.N)
            for c in self.C
        }
        for c in self.C:
            if alpha is None and self.tau[c] == 0:
                self.alpha[random.choice(list(self.N)), c] = 1
                self.tau[c] = 1
            elif alpha is not None and self.tau[c] == 0:
                raise ValueError(f'No sites with characteristic {c}.')
            else:
                pass

    def print(self):
        print(f'N = {len(self.N)} (sites)')
        print(f'K = {len(self.K)} (vehicles)')
        print(f'T_max = {self.T_max} (max time)')
        print(f'C = {len(self.C)} (characteristics)\n')

    def validate_objective(self, obj: float, exception: bool = False):
        if self.optimal_value is None:
            print(f'Optimal value is not known, cannot validate given objective.')
            return
        if abs(obj - self.optimal_value) > 1e-6:
            if exception:
                raise ValueError(f'Real optimal value is {self.optimal_value}, but got {obj}.')
            else:
                warnings.warn(f'Real optimal value is {self.optimal_value}, but got {obj}.')
        else:
            print(f'Objective value validated to be optimal.')


class Formulation(ABC):
    def __init__(self, instance: Instance, activations: Dict[str, bool] = None):
        self.instance = instance
        self.activations = activations if activations is not None else {}
        self.solver = gp.Model()
        self.constraints = {}
        self.callback = None

    @abstractmethod
    def define_variables(self):
        """
        Define the variables of the formulation.
        """
        pass

    @abstractmethod
    def fill_constraints(self):
        """
        Fill the dictionary self.constraints with the constraints of the formulation.
        """
        pass

    @abstractmethod
    def define_objective(self):
        """
        Define the objective function of the formulation.
        """
        pass

    @abstractmethod
    def build_solution(self) -> 'Solution':
        """
        Build the solution from the variables of the formulation. Store it in self.solution.

        Returns:
            object: 
        """
        pass

    def define_constraints(self):
        """
        Define the constraints of the formulation according to the activation dictionary.
        """
        self.fill_constraints()
        assert not set(self.activations.keys()) - set(self.constraints.keys()), \
            f'Some activations refer to non-existent constraints: ' \
            f'{set(self.activations.keys()) - set(self.constraints.keys())}.'
        print(f'Constraints:')
        for constraint_name, active in self.constraints.items():
            if self.activations.get(constraint_name, True):
                print(f' - {constraint_name}')
                self.constraints[constraint_name]()

    def formulate(self):
        """
        Formulate the problem. After this, the solver can solve the problem.
        """
        self.define_variables()
        self.define_constraints()
        self.define_objective()


class Solution:
    """
    This class represents a solution of a formulation.
    A solution of the problem is represented with the set of binary variables x[i, j, k] = 1 if the vehicle k
    travels from i to j.
    """

    def __init__(self, inst: Instance, x: Dict[Tuple[int, int, int], int], obj: float):
        self.inst = inst
        self.x = x
        self.obj = obj

        self.y = self.calculate_y()
        self.coverage_ratios = self.calculate_coverage_ratios()
        self.Wp = self.calculate_Wp()
        self.m = self.calculate_m()
        self.routes = self.calculate_routes()

        self.check_obj()

    def check_obj(self):
        """
        Check if the objective provided by the solver matches the objective calculated using the solution.
        """
        calculated_obj = min(self.coverage_ratios.values())
        assert abs(self.obj - calculated_obj) < 1e-6, \
            f'Objective provided by the solver ({self.obj}) does not match the objective ' \
            f'calculated using the solution ({calculated_obj}).'
        self.obj = calculated_obj

    def calculate_routes(self) -> Dict[int, Tuple[List[int], float]]:
        """
        Calculate the routes of the solution, and its duration.
        """
        routes = {}
        for k in self.inst.K:
            route = self.find_route(k)
            routes[k] = (route, self.find_duration(route))
        return routes

    def find_duration(self, route) -> float:
        """
        Find the length of the route.
        """
        return sum(self.inst.t[i, j] for i, j in zip(route[:-1], route[1:]))

    def calculate_coverage_ratios(self) -> Dict[str, float]:
        """
        Calculate the coverage ratios of the solution.
        """
        return {
            c: sum(self.inst.alpha[i, c] * self.y[i, k] for i in self.inst.N for k in self.inst.K) / self.inst.tau[c]
            for c in self.inst.C
        }

    def calculate_y(self) -> Dict[Tuple[int, int], int]:
        """
        Calculate the number of times each site is visited.
        """
        return {
            (i, k): sum(self.x[i, j, k] for j in self.inst.N_0)
            for i in self.inst.N_0
            for k in self.inst.K
        }

    def calculate_Wp(self) -> float:
        """
        Calculate the duration of the routes.
        """
        return sum(
            self.inst.t[i, j] * self.x[i, j, k]
            for i in self.inst.N_0
            for j in self.inst.N_0
            for k in self.inst.K
        )

    def calculate_m(self) -> int:
        """
        Calculate the number of nodes visited.
        """
        return sum(
            self.x[0, j, k]
            for j in self.inst.N_0
            for k in self.inst.K
        )

    def print(self, verbose: int = 1) -> None:
        """
        Print the solution. If verbose = 2, print the coverage ratios of the solution.

        Args:
            verbose: if 1, print the solution. If 2, print the solution and the coverage ratios.
        """
        print(f"{'-' * 30}")
        print(f"{'-' * 10} Solution {'-' * 10}")
        print(f"{'-' * 30}")
        print(f'Objective: {round(self.obj, 3)}')
        print(f'Route duration (Wp): {round(self.Wp, 1)}')
        print()

        print(' --Routes:')
        for k, (route, time) in self.routes.items():
            print(f'Route of vehicle {k}: {route} -> {round(time, 1)} time units')
        if verbose == 2:
            print(' --Coverage ratios:')
            for c in self.inst.C:
                print(f'CR of characteristic {c}: {self.coverage_ratios[c]}\n'
                      f'  Represented by {set(i for i in self.inst.N if self.inst.alpha[i, c] == 1)}\n'
                      f'  Covered by {set(i for i in self.inst.N if self.inst.alpha[i, c] == 1 and sum(self.y[i, k] for k in self.inst.K) >= 1)}')
            print(' --Characteristics distribution:')
            for i in self.inst.N:
                print(f"Site {i}: {set(c for c in self.inst.C if self.inst.alpha[i, c] == 1)}"
                      f"{' -> Selected' if sum(self.y[i, k] for k in self.inst.K) >= 1 else ''}")

        print(f"{'-' * 30}")
        print(f"{'-' * 30}")

    def find_route(self, k) -> List[int]:
        """
        Find the route of vehicle k.

        Args:
            k: vehicle index

        Returns:
            Route of vehicle k.
        """
        route = [0]
        while True:
            j = self.find_next(route[-1], k, route)
            if not j:
                return route + [0]
            route.append(j)

    def find_next(self, i: int, k: int, route: List[int]) -> Optional[int]:
        """
        Find the next node in the route.

        Args:
            i: current node
            k: vehicle index
            route: current route

        Returns:
            Next node in the route. If there is no next node (i.e. the route is complete), return None.
        """
        for j in self.inst.N_0:
            if self.x[i, j, k] == 1 and j not in route:
                return j
        return None

    def routes_to_string(self) -> str:
        """
        Return a string representation of the routes.
        """
        return ' // '.join(f'Route of vehicle {k}: {route}' for k, route in self.routes.items())

    def draw(self):
        """
        Draw the solution using networkx.
        """
        g = nx.DiGraph()
        g.add_nodes_from(self.inst.N)
        g.add_edges_from((i, j) for i, j, k in self.x if self.x[i, j, k] == 1)

        pos = nx.spring_layout(g)
        nx.draw(g, pos, with_labels=True, node_color="skyblue", node_size=1000, font_size=10)

        plt.show()
