from src.config import Config
from utils import Formulation, Solution
import gurobipy as gp
import pandas as pd
from datetime import datetime


class Optimizer:
    def __init__(self, formulation: Formulation, config: Config):
        self.formulation = formulation
        self.solver = formulation.solver
        self.config = config
        self.solver.setParam('TimeLimit', self.config.time_limit * 60)

    def run(self):
        self.formulation.formulate()
        self.solver.optimize(self.formulation.callback)
        print(f'Solve time: {self.solver.Runtime} seconds')
        print(f'Status: {self.solver.status}')

        if self.solver.status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.ITERATION_LIMIT, gp.GRB.NODE_LIMIT]:
            if self.solver.status == gp.GRB.OPTIMAL:
                self.formulation.instance.validate_objective(self.solver.objVal)
            solution = self.formulation.build_solution()
            if self.config.print_solution:
                solution.print(self.config.print_solution)
            if self.config.draw_solution:
                solution.draw()
            self.save_results(solution)
        elif self.solver.status in [gp.GRB.INFEASIBLE, gp.GRB.INF_OR_UNBD]:
            self.infeasibility_analysis()
            return
        else:
            raise Exception(f'Unexpected solver status: {self.solver.status}')

    def save_results(self, solution: Solution):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        results = {
            'name': self.formulation.instance.name,
            'type': self.config.instance_type,
            'id': self.formulation.instance.id,
            'formulation': self.config.formulation,
            'N': len(self.formulation.instance.N),
            'K': len(self.formulation.instance.K),
            'T_max': self.formulation.instance.T_max,
            'C': len(self.formulation.instance.C),
            'solve_time': self.solver.Runtime,
            'status': self.solver.status,
            'time_limit': self.config.time_limit,
            'objective': self.solver.objVal,
            'best_bound': self.solver.ObjBound,
            'best_int': self.solver.ObjBoundC,
            'Wp': solution.Wp,
            'm': solution.m,
            'gap': self.solver.MIPGap,
            'n_vars': self.solver.NumVars,
            'n_cons': self.solver.NumConstrs,
            'n_nodes': self.solver.NodeCount,
            'n_solutions': self.solver.SolCount,
            'routes': solution.routes_to_string(),
            'timestamp': timestamp,
        }

        try:
            results_df = pd.read_csv(self.config.results_file, sep=';', decimal='.')
        except FileNotFoundError:
            results_df = pd.DataFrame(columns=list(results.keys()))

        results_df.loc[len(results_df)] = list(results.values())
        results_df.to_csv(self.config.results_file, index=False, sep=';', decimal='.')

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