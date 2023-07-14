from utils import Formulation
import gurobipy as gp


class Optimizer:
    def __init__(self, formulation: Formulation):
        self.formulation = formulation
        self.solver = formulation.solver

    def run(self):
        self.formulation.formulate()
        self.solver.optimize()

        if self.solver.status == gp.GRB.OPTIMAL:
            solution = self.formulation.build_solution()
            self.formulation.instance.print()
            solution.print()
