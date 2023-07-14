from src.config import Config
from utils import Formulation
import gurobipy as gp


class Optimizer:
    def __init__(self, formulation: Formulation, config: Config):
        self.formulation = formulation
        self.solver = formulation.solver
        self.solver.setParam('TimeLimit', config.time_limit * 60)

    def run(self):
        self.formulation.formulate()
        self.solver.optimize()
        print(f'Solve time: {self.solver.Runtime} seconds')

        if self.solver.status == gp.GRB.OPTIMAL:
            solution = self.formulation.build_solution()
            self.formulation.instance.print()
            solution.print()
