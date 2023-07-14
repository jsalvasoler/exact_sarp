import random
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union
import gurobipy as gp


class Instance:
    def __init__(self, N_size, K_size, T_max, C_size, seed=None):
        if seed is not None:
            random.seed(seed)

        self.N = {i + 1 for i in range(N_size)}
        self.K = {i + 1 for i in range(K_size)}
        self.N_0 = {0}.union(self.N)
        self.t = {
            (i, j): 0 if i == j else random.randint(1, 10)
            for i in self.N_0
            for j in self.N_0
        }
        self.T_max = T_max
        self.C = {i + 1 for i in range(C_size)}
        self.alpha = {
            (i, c): 1 if random.random() < 0.25 else 0
            for c in self.C
            for i in self.N
        }
        self.tau = {
            c: sum(self.alpha[i, c] for i in self.N)
            for c in self.C
        }
        for c in self.C:
            if self.tau[c] == 0:
                self.alpha[random.choice(list(self.N)), c] = 1
                self.tau[c] = 1

    def print(self):
        print(f'N = {self.N} (sites)')
        print(f'K = {self.K} (vehicles)')
        print(f'C = {self.C} (characteristics)')
        print()


class Formulation(ABC):
    def __init__(self, instance: Instance, activations: Dict[str, bool]):
        self.instance = instance
        self.activations = activations
        self.solver = gp.Model()
        self.constraints = {}

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
        """
        pass

    def define_constraints(self):
        """
        Define the constraints of the formulation according to the activation dictionary.
        """
        self.fill_constraints()
        assert not set(self.activations.keys()) - set(self.constraints.keys()), \
            f'Some activations refer to non-existent constraints.'
        for constraint_name, active in self.constraints.items():
            if self.activations.get(constraint_name, True):
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
        self.coverage_ratios = None
        self.y = None

        self.check_obj()

    def check_obj(self):
        """
        Check if the objective provided by the solver matches the objective calculated using the solution.
        """
        self.y = {
            (i, k): sum(self.x[i, j, k] for j in self.inst.N_0)
            for i in self.inst.N_0
            for k in self.inst.K
        }
        self.coverage_ratios = {
            c: sum(self.inst.alpha[i, c] * self.y[i, k] for i in self.inst.N for k in self.inst.K) / self.inst.tau[c]
            for c in self.inst.C
        }
        calculated_obj = min(self.coverage_ratios.values())
        assert abs(self.obj - calculated_obj) < 1e-6, \
            f'Objective provided by the solver ({self.obj}) does not match the objective ' \
            f'calculated using the solution ({calculated_obj}).'
        self.obj = calculated_obj

    def print(self):
        print(f"{'-' * 30}")
        print(f"{'-' * 10} Solution {'-' * 10}")
        print(f"{'-' * 30}")
        print(f'Objective: {self.obj}')

        print(' --Routes:')
        for k in self.inst.K:
            route = self.find_route(k)
            print(f'Route of vehicle {k}: {route}')
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

    def find_route(self, k):
        route = [0]
        while True:
            j = self.find_next(route[-1], k, route)
            if not j:
                return route + [0]
            route.append(j)

    def find_next(self, i, k, route):
        for j in self.inst.N_0:
            if self.x[i, j, k] == 1 and j not in route:
                return j
        return None
