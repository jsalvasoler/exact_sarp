import gurobipy as gp

from src.utils import Formulation, Instance, Solution


# noinspection DuplicatedCode
class MTZOptFormulation(Formulation):
    def __init__(
        self, inst: Instance, activations: dict = None, linear_relax: bool = False
    ):
        super().__init__(inst, activations, linear_relax)
        self.name = "mtz_opt"

        self.x = {}
        self.y = {}
        self.u = {}
        self.z = None

    def define_variables(self):
        var_type = gp.GRB.CONTINUOUS if self.linear_relax else gp.GRB.BINARY
        for i in self.instance.N:
            self.y[i] = self.solver.addVar(vtype=var_type, name=f"y_{i}", lb=0, ub=1)
        for i in self.instance.N_0:
            for j in self.instance.N_0:
                self.x[i, j] = self.solver.addVar(
                    vtype=var_type, name=f"x_{i}_{j}", lb=0, ub=1
                )
        for i in self.instance.N_0:
            self.u[i] = self.solver.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"u_{i}", lb=0, ub=self.instance.T_max
            )
        self.z = self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name="z", lb=0, ub=1)

    def constraint_define_obj(self):
        for c in self.instance.C:
            self.solver.addConstr(
                self.z * self.instance.tau[c]
                <= gp.quicksum(
                    self.instance.alpha[i, c] * self.y[i] for i in self.instance.N
                ),
                name=f"define_obj_{c}",
            )

    def constraint_leave(self):
        for i in self.instance.N:
            self.solver.addConstr(
                gp.quicksum(self.x[i, j] for j in self.instance.N_0) == self.y[i],
                name=f"leave_{i}",
            )

    def constraint_enter(self):
        for i in self.instance.N:
            self.solver.addConstr(
                gp.quicksum(self.x[j, i] for j in self.instance.N_0) == self.y[i],
                name=f"enter_{i}",
            )

    def constraint_number_of_vehicles_enter(self):
        self.solver.addConstr(
            gp.quicksum(self.x[0, j] for j in self.instance.N) == len(self.instance.K),
            name="number_of_vehicles_enter",
        )

    def constraint_number_of_vehicles_leave(self):
        self.solver.addConstr(
            gp.quicksum(self.x[j, 0] for j in self.instance.N) == len(self.instance.K),
            name="number_of_vehicles_leave",
        )

    def constraint_mtz(self):
        for i in self.instance.N:
            for j in self.instance.N_0:
                if i == j:
                    continue
                self.solver.addConstr(
                    self.u[j] - self.u[i]
                    >= self.x[i, j] * (self.instance.T_max + self.instance.t[i, j])
                    - self.instance.T_max,
                    name=f"mtz_{i}_{j}",
                )

    def constraint_not_stay(self):
        for i in self.instance.N:
            self.solver.addConstr(self.x[i, i] == 0, name=f"not_stay_{i}")

    def constraint_arrival_time(self):
        for i in self.instance.N:
            self.solver.addConstr(self.u[0] >= self.u[i], name=f"arrival_time_{i}")

    def constraint_u_lower_bound(self):
        for i in self.instance.N_0:
            self.solver.addConstr(
                self.u[i] >= self.instance.t[0, i] * self.x[0, i],
                name=f"u_lower_bound_{i}",
            )

    def fill_constraints(self):
        # Get constraint names by looking at attributes (methods) with prefix 'constraint_'
        prefix = "constraint_"
        constraint_names = [
            method_name[len(prefix) :]
            for method_name in dir(self)
            if method_name.startswith(prefix)
        ]

        for constraint_name in constraint_names:
            self.constraints[constraint_name] = getattr(
                self, f"{prefix}{constraint_name}"
            )

    def define_objective(self):
        self.solver.setObjective(self.z, gp.GRB.MAXIMIZE)

    def build_solution(self) -> Solution:
        x = {
            (i, j, k): 0
            for i in self.instance.N_0
            for j in self.instance.N_0
            for k in self.instance.K
        }
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

    def impose_solution(self, x, y, u):
        for k in self.x.keys():
            assert k in x, f"{k} not in the provided x values"
            self.solver.addConstr(self.x[k] == x[k])
        for k in self.y.keys():
            assert k in y, f"{k} not in the provided y values"
            self.solver.addConstr(self.y[k] == y[k])
        for k in self.u.keys():
            assert k in u, f"{k} not in the provided u values"
            self.solver.addConstr(self.u[k] == u[k])
