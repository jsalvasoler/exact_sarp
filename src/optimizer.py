from src.config import Config
from utils import Formulation
import gurobipy as gp


class Optimizer:
    def __init__(self, formulation: Formulation, config: Config):
        self.formulation = formulation
        self.solver = formulation.solver
        self.config = config
        self.solver.setParam('TimeLimit', self.config.time_limit * 60)

    def run(self):
        self.formulation.formulate()
        self.solver.optimize()
        print(f'Solve time: {self.solver.Runtime} seconds')
        print(f'Status: {self.solver.status}')

        if self.solver.status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.ITERATION_LIMIT, gp.GRB.NODE_LIMIT]:
            solution = self.formulation.build_solution()
            if self.config.print_solution:
                solution.print(self.config.print_solution)
        elif self.solver.status == gp.GRB.INFEASIBLE:
            self.infeasibility_analysis()
        else:
            pass

    def infeasibility_analysis(self):
        self.solver.computeIIS()
        for c in self.solver.getConstrs():
            if c.IISConstr:
                print(f'\t{c.constrname}: {self.solver.getRow(c)} {c.Sense} {c.RHS}')

        for v in self.solver.getVars():
            if v.IISLB:
                print(f'\t{v.varname} ≥ {v.LB}')
            if v.IISUB:
                print(f'\t{v.varname} ≤ {v.UB}')

