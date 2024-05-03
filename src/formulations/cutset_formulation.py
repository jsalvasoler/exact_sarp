import gurobipy as gp
import networkx as nx
import itertools

from utils import Formulation, Instance, Solution


# noinspection DuplicatedCode
class CutSetFormulation(Formulation):
    def __init__(
        self,
        inst: Instance,
        activations: dict = None,
        linear_relax: bool = False,
        full_model: bool = False,
    ):
        super().__init__(inst, activations, linear_relax)
        self.name = "cutset"

        self.x = {}
        self.y = {}
        self.z = None

        self.full_model = full_model

        if not self.full_model:
            self.activations["cutset"] = False
            self.callback = create_callback(activations)
            self.solver.Params.lazyConstraints = 1
        else:
            self.activations["cutset"] = True

    def fill_constraints(self):
        # Get constraint names by looking at attributes (methods) with prefix 'constraint_'
        prefix = "constraint_"
        constraint_names = [
            method_name[len(prefix) :]
            for method_name in dir(self)
            if method_name.startswith(prefix)
        ]
        if self.full_model:
            constraint_names.append("cutset")

        for constraint_name in constraint_names:
            self.constraints[constraint_name] = getattr(
                self, f"{prefix}{constraint_name}"
            )

    def define_objective(self):
        self.solver.setObjective(self.z, gp.GRB.MAXIMIZE)

    def build_solution(self) -> Solution:
        x = {
            (i, j, k): 1 if self.x[i, j, k].X > 0.5 else 0
            for i in self.instance.N_0
            for j in self.instance.N_0
            for k in self.instance.K
        }
        return Solution(self.instance, x, self.solver.objVal)

    def define_variables(self):
        var_type = gp.GRB.CONTINUOUS if self.linear_relax else gp.GRB.BINARY
        for i in self.instance.N_0:
            for k in self.instance.K:
                self.y[i, k] = self.solver.addVar(
                    vtype=var_type, name=f"y_{i}_{k}", lb=0, ub=1
                )
        for i in self.instance.N_0:
            for j in self.instance.N_0:
                for k in self.instance.K:
                    self.x[i, j, k] = self.solver.addVar(
                        vtype=var_type, name=f"x_{i}_{j}_{k}", lb=0, ub=1
                    )
        self.z = self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name="z", lb=0, ub=1)

        self.solver._x = self.x
        self.solver._y = self.y
        self.solver._z = self.z

    def constraint_cutset(self):
        # Use itertools to find all subsets of N
        subsets = []
        for i in range(1, len(self.instance.N)):
            subsets.extend(itertools.combinations(self.instance.N, i))
        for k in self.instance.K:
            for subset in subsets:
                for h in subset:
                    delta = {
                        (i, j)
                        for (i, j, k) in self.x.keys()
                        if i in subset and j not in subset
                    }
                    self.solver.addConstr(
                        gp.quicksum(self.x[i, j, k] for (i, j) in delta)
                        >= self.y[h, k],
                        name=f"cutset_{k}_{subset}_{h}",
                    )

    def constraint_define_obj(self):
        for c in self.instance.C:
            self.solver.addConstr(
                self.z * self.instance.tau[c]
                <= gp.quicksum(
                    self.instance.alpha[i, c] * self.y[i, k]
                    for i in self.instance.N
                    for k in self.instance.K
                ),
                name=f"define_obj_{c}",
            )

    def constraint_leave(self):
        for i in self.instance.N_0:
            for k in self.instance.K:
                self.solver.addConstr(
                    gp.quicksum(self.x[i, j, k] for j in self.instance.N_0)
                    == self.y[i, k],
                    name=f"leave_{i}_{k}",
                )

    def constraint_enter(self):
        for i in self.instance.N_0:
            for k in self.instance.K:
                self.solver.addConstr(
                    gp.quicksum(self.x[j, i, k] for j in self.instance.N_0)
                    == self.y[i, k],
                    name=f"enter_{i}_{k}",
                )

    def constraint_visit(self):
        for i in self.instance.N:
            self.solver.addConstr(
                gp.quicksum(self.y[i, k] for k in self.instance.K) <= 1,
                name=f"visit_{i}",
            )

    def constraint_number_of_vehicles_hard(self):
        self.solver.addConstr(
            gp.quicksum(self.y[0, k] for k in self.instance.K) == len(self.instance.K),
            name="number_of_vehicles_hard",
        )

    def constraint_enter_depot(self):
        self.solver.addConstr(
            gp.quicksum(
                self.x[0, i, k] for k in self.instance.K for i in self.instance.N
            )
            == gp.quicksum(
                self.x[i, 0, k] for k in self.instance.K for i in self.instance.N
            ),
            name="enter_depot",
        )

    def constraint_max_time(self):
        for k in self.instance.K:
            self.solver.addConstr(
                gp.quicksum(
                    self.instance.t[i, j] * self.x[i, j, k]
                    for i in self.instance.N_0
                    for j in self.instance.N_0
                )
                <= self.instance.T_max,
                name=f"max_time_{k}",
            )

    def constraint_not_stay(self):
        for i in self.instance.N:
            for k in self.instance.K:
                self.solver.addConstr(self.x[i, i, k] == 0, name=f"not_stay_{i}_{k}")


def find_min_cut(x, y):
    keys = list(x.keys())
    K = set([k for _, _, k in keys])
    N = set([i for i, _, _ in keys])
    for k in K:
        g = nx.DiGraph()
        g.add_weighted_edges_from(
            [(i, j, value) for (i, j, k_), value in x.items() if k_ == k]
        )
        for i in N - {0}:
            flow = nx.maximum_flow_value(g, 0, i, capacity="weight")
            if flow < y[i, k] - 0.0001:
                val, partition = nx.minimum_cut(g, 0, i, capacity="weight")
                outgoing_edges = [
                    (i, j)
                    for i, j in g.edges
                    if i in partition[0] and j in partition[1]
                ]
                yield outgoing_edges, i, k


def add_cutset_constraint(activations: dict, model: gp.Model, where):
    if activations["cutset_integer"]:
        if where == gp.GRB.Callback.MIPSOL:
            x = model.cbGetSolution(model._x)
            y = model.cbGetSolution(model._y)
            add_cut_to_formulation(model, x, y)
    if activations["cutset_relaxation"]:
        if (
            where == gp.GRB.Callback.MIPNODE
            and model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) == gp.GRB.OPTIMAL
        ):
            x = model.cbGetNodeRel(model._x)
            y = model.cbGetNodeRel(model._y)
            add_cut_to_formulation(model, x, y)


def create_callback(activations: dict):
    def partial_func(model, where):
        add_cutset_constraint(activations, model, where)

    return partial_func


def add_cut_to_formulation(model: gp.Model, x, y):
    gen = find_min_cut(x, y)
    for outgoing_edges, node, k in gen:
        model._num_lazy_constraints_added += 1
        model.cbLazy(
            gp.quicksum(model._x[i, j, k] for i, j in outgoing_edges)
            >= model._y[node, k]
        )
