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
            (i, c): random.randint(0, 1)
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

        self.check_obj()

    def check_obj(self):
        """
        Check if the objective provided by the solver matches the objective calculated using the solution.
        """
        y = {
            (i, k): sum(self.x[i, j, k] for j in self.inst.N_0)
            for i in self.inst.N_0
            for k in self.inst.K
        }
        coverage_ratios = {
            c: sum(self.inst.alpha[i, c] * y[i, k] for i in self.inst.N for k in self.inst.K) / self.inst.tau[c]
            for c in self.inst.C
        }
        calculated_obj = min(coverage_ratios.values())
        assert abs(self.obj - calculated_obj) < 1e-6, \
            f'Objective provided by the solver ({self.obj}) does not match the objective ' \
            f'calculated using the solution ({calculated_obj}).'

    def print(self):
        print('Solution:')
        for i in self.inst.N_0:
            for j in self.inst.N_0:
                for k in self.inst.K:
                    if self.x[i, j, k] == 1:
                        print(f'x[{i}, {j}, {k}] = 1')
        print(f'Objective: {self.obj}')
