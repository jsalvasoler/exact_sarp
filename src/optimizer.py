from src.config import Config
from utils import Formulation, Solution
import gurobipy as gp
import pandas as pd
from datetime import datetime
import warnings


class Optimizer:
    def __init__(self, formulation: Formulation, config: Config):
        self.formulation = formulation
        self.solver = formulation.solver
        self.config = config
        self.solver.setParam('TimeLimit', self.config.time_limit * 60)
        self.solver.setParam('Threads', 12)

    def run(self):
        self.formulation.formulate()
        self.solver.optimize(self.formulation.callback)
        print(f'Solve time: {self.solver.Runtime} seconds')
        print(f'Status: {self.solver.status}')

        if self.solver.status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.ITERATION_LIMIT, gp.GRB.NODE_LIMIT]:
            if self.solver.status == gp.GRB.OPTIMAL and self.config.instance_type == 'small':
                self.formulation.instance.validate_objective_for_small_instances(
                    self.solver.objVal, self.config.exception_when_non_optimal)
            solution = self.formulation.build_solution()
            if self.config.print_solution:
                solution.print(self.config.print_solution)
            if self.config.draw_solution:
                solution.draw()
            self.save_results(solution)
        elif self.solver.status in [gp.GRB.INFEASIBLE]:
            self.infeasibility_analysis()
            return
        elif self.solver.status in [gp.GRB.INF_OR_UNBD]:
            return
        else:
            raise ValueError(f'Unexpected solver status: {self.solver.status}')

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
            'network_type': self.formulation.instance.network_type,
            'm_solution': solution.m,
            'solve_time': self.solver.Runtime,
            'status': self.solver.status,
            'time_limit': self.config.time_limit,
            'objective': self.solver.objVal,
            'best_bound': self.solver.ObjBound,
            'best_int': self.solver.ObjBoundC,
            'mip_gap': self.solver.MIPGap,
            'Wp': solution.Wp,
            'gap': self.solver.MIPGap,
            'n_vars': self.solver.NumVars,
            'n_cons': self.solver.NumConstrs,
            'n_nodes': self.solver.NodeCount,
            'n_added_cuts': self.solver._num_lazy_constraints_added,
            'n_solutions': self.solver.SolCount,
            'routes': solution.routes_to_string(),
            'timestamp': timestamp,
        }
        results = {**results, **self.formulation.instance.instance_results}

        try:
            results_df = pd.read_csv(self.config.results_file, sep=';', decimal=',')
        except FileNotFoundError:
            results_df = pd.DataFrame(columns=list(results.keys()))

        results_df.loc[len(results_df)] = list(results.values())
        try:
            results_df.to_csv(self.config.results_file, index=False, sep=';', decimal=',')
        except PermissionError:
            warnings.warn('Could not save results. Saving it with a timestamp.')
            results_df.to_csv(f'results_{timestamp}.csv', index=False, sep=';', decimal=',')

    def infeasibility_analysis(self):
        warnings.warn('Model is infeasible. Performing infeasibility analysis.')
        self.solver.computeIIS()
        for c in self.solver.getConstrs():
            if c.IISConstr:
                print(f'\t{c.constrname}: {self.solver.getRow(c)} {c.Sense} {c.RHS}')

        for v in self.solver.getVars():
            if v.IISLB:
                print(f'\t{v.varname} >= {v.LB}')
            if v.IISUB:
                print(f'\t{v.varname} =< {v.UB}')
