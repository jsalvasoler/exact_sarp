import gurobipy as gp

from src.utils import Formulation, Instance, Solution


class SCFFormulation(Formulation):
    def __init__(self, inst: Instance, activations: dict = None):
        super().__init__(inst, activations)
        self.f = {}     # keys are (i,j) and values are the flow i->j
        self.x = {}     # keys are (i,j) and is 1 if edge (i,j) is used (has flow > 0)
        self.y = {}     # keys are (i) and is 1 if node i is visited
        self.z = None

    def define_variables(self):
        for i in self.instance.N_0:
            self.y[i] = self.solver.addVar(vtype=gp.GRB.BINARY, name=f'y_{i}', lb=0, ub=1)
        for i in self.instance.N_0:
            for j in self.instance.N_0:
                self.f[i, j] = self.solver.addVar(
                    vtype=gp.GRB.CONTINUOUS, name=f'f_{i}_{j}', lb=0, ub=self.instance.T_max)
                self.x[i, j] = self.solver.addVar(
                    vtype=gp.GRB.BINARY, name=f'x_{i}_{j}', lb=0, ub=1)
        self.z = self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name='z', lb=0, ub=gp.GRB.INFINITY)

    def constraint_define_obj(self):
        for c in self.instance.C:
            self.solver.addConstr(
                self.z * self.instance.tau[c] <= gp.quicksum(
                    self.instance.alpha[i, c] * self.y[i] for i in self.instance.N),
                name=f'define_obj_{c}'
            )

    def constraint_link_flow_and_edges(self):
        for i in self.instance.N_0:
            for j in self.instance.N_0:
                self.solver.addConstr(
                    self.f[i, j] <= self.instance.T_max * self.x[i, j],
                    name=f'link_flow_and_edges_1_{i}_{j}'
                )
                # self.solver.addConstr(
                #     self.x[i, j] <= 1e6 * self.f[i, j],
                #     name=f'link_flow_and_edges_2_{i}_{j}'
                # )

    def constraint_leave_depot_route(self):     # K routes must leave the depot (hard constraint)
        self.solver.addConstr(
            gp.quicksum(self.x[0, i] for i in self.instance.N) == len(self.instance.K),
            name=f'leave_depot_route'
        )

    def constraint_enter_depot_route(self):     # K routes must enter the depot
        self.solver.addConstr(
            gp.quicksum(self.x[i, 0] for i in self.instance.N) == len(self.instance.K),
            name=f'enter_depot_route'
        )

    def constraint_enter_depot_flow(self):      # The flow entering the depot must be equal to the cost of the edge
        for i in self.instance.N:
            self.solver.addConstr(
                self.f[i, 0] == self.x[i, 0] * self.instance.t[i, 0],
                name=f'enter_depot_flow_{i}'
            )

    def constraint_flow_visit(self):
        for i in self.instance.N:
            self.solver.addConstr(
                gp.quicksum(self.f[j, i] - self.f[i, j] for j in self.instance.N_0) ==
                gp.quicksum(self.x[j, i] * self.instance.t[j, i] for j in self.instance.N_0),
                name=f'flow_visit_{i}'
            )

    def constraint_route_visit(self):
        for i in self.instance.N:
            self.solver.addConstr(
                gp.quicksum(self.x[j, i] + self.x[i, j] for j in self.instance.N_0) == 2 * self.y[i],
                name=f'route_visit_{i}'
            )

    def constraint_route_visit_hard(self):
        for i in self.instance.N:
            self.solver.addConstr(
                gp.quicksum(self.x[j, i] for j in self.instance.N_0) == self.y[i],
                name=f'route_visit_hard_in_{i}'
            )
            self.solver.addConstr(
                gp.quicksum(self.x[i, j] for j in self.instance.N_0) == self.y[i],
                name=f'route_visit_hard_out_{i}'
            )

    def constraint_not_stay(self):
        for i in self.instance.N:
            self.solver.addConstr(
                self.x[i, i] == 0,
                name=f'not_stay_{i}'
            )

    def fill_constraints(self):
        # Get constraint names by looking at attributes (methods) with prefix 'constraint_'
        prefix = 'constraint_'
        constraint_names = [method_name[len(prefix):] for method_name in dir(self) if method_name.startswith(prefix)]

        for constraint_name in constraint_names:
            self.constraints[constraint_name] = getattr(self, f'{prefix}{constraint_name}')

    def define_objective(self):
        self.solver.setObjective(self.z, gp.GRB.MAXIMIZE)

    def build_solution(self) -> Solution:
        x = {(i, j, k): 0 for i in self.instance.N_0 for j in self.instance.N_0 for k in self.instance.K}
        starting_nodes = [i for i in self.instance.N if self.x[0, i].X > 0.5]
        k = 1
        while k <= len(self.instance.K):
            i = starting_nodes[k - 1]
            x[0, i, k] = 1
            while i != 0:
                for j in self.instance.N_0:
                    if self.x[i, j].X > 0.5:
                        x[i, j, k] = 1
                        i = j
                        break
            k += 1
        return Solution(self.instance, x, self.solver.objVal)
