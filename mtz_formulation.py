from abc import ABC
import gurobipy as gp

from utils import Formulation, Instance, Solution


class MTZFormulation(Formulation):
    def __init__(self, inst: Instance, activations: dict = None):
        super().__init__(inst, activations)
        self.x = {}
        self.y = {}
        self.u = {}
        self.z = None

        self.constraint_names = [
            'define', 'leave', 'enter', 'visit', 'leave_depot', 'max_time'
        ]

    def define_variables(self):
        for i in self.instance.N_0:
            for k in self.instance.K:
                self.y[i, k] = self.solver.addVar(vtype=gp.GRB.BINARY, name=f'y_{i}_{k}', lb=0, ub=1)
        for i in self.instance.N_0:
            for j in self.instance.N_0:
                for k in self.instance.K:
                    self.x[i, j, k] = self.solver.addVar(vtype=gp.GRB.BINARY, name=f'x_{i}_{j}_{k}', lb=0, ub=1)
        for i in self.instance.N:
            self.u[i] = self.solver.addVar(vtype=gp.GRB.INTEGER, name=f'u_{i}', lb=0,
                                           ub=len(self.instance.N) - 1)
        self.z = self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name='z', lb=0, ub=gp.GRB.INFINITY)

    def constraint_define(self):
        for c in self.instance.C:
            self.solver.addConstr(
                self.z * self.instance.tau[c] <= gp.quicksum(self.instance.alpha[i, k] * self.y[i, k]
                                                             for i in self.instance.N for k in self.instance.K),
                name=f'define_z{c}'
            )

    def constraint_leave(self):
        for i in self.instance.N_0:
            for k in self.instance.K:
                self.solver.addConstr(
                    gp.quicksum(self.x[i, j, k] for j in self.instance.N_0) == self.y[i, k],
                    name=f'leave_{i}_{k}'
                )

    def constraint_enter(self):
        for i in self.instance.N_0:
            for k in self.instance.K:
                self.solver.addConstr(
                    gp.quicksum(self.x[j, i, k] for j in self.instance.N_0) == self.y[i, k],
                    name=f'enter_{i}_{k}'
                )

    def constraint_visit(self):
        for i in self.instance.N:
            self.solver.addConstr(
                gp.quicksum(self.y[i, k] for k in self.instance.K) <= 1,
                name=f'visit_{i}'
            )

    def constraint_leave_depot(self):
        self.solver.addConstr(
            gp.quicksum(self.y[0, k] for k in self.instance.K) <= len(self.instance.K),
            name='leave_depot'
        )

    def constraint_max_time(self):
        for k in self.instance.K:
            self.solver.addConstr(
                gp.quicksum(self.instance.t[i, j] * self.x[i, j, k]
                            for i in self.instance.N_0 for j in self.instance.N_0) <= self.instance.T_max,
                name=f'max_time_{k}'
            )

    def constraint_mtz(self):
        for i in self.instance.N:
            for j in self.instance.N:
                if i == j:
                    continue
                for k in self.instance.K:
                    self.solver.addConstr(
                        self.u[i] - self.u[j] + len(self.instance.N) * (self.x[i, j, k]) <= len(self.instance.N) - 1,
                        name=f'mtz_{i}_{j}_{k}'
                    )

    def fill_constraints(self):
        for constraint_name in self.constraint_names:
            self.constraints[constraint_name] = getattr(self, f'constraint_{constraint_name}')

    def define_objective(self):
        self.solver.setObjective(self.z, gp.GRB.MAXIMIZE)

    def build_solution(self) -> Solution:
        x = {
            (i, j, k): self.x[i, j, k].X
            for i in self.instance.N_0
            for j in self.instance.N_0
            for k in self.instance.K
        }
        return Solution(self.instance, x, self.solver.objVal)
