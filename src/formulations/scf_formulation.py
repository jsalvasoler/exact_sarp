import gurobipy as gp
import itertools
import networkx as nx

from heuristics.greedy_heuristic import GreedyHeuristic
from src.utils import Formulation, Instance, Solution


class SCFFormulation(Formulation):
    def __init__(
        self,
        inst: Instance,
        activations: dict = None,
        linear_relax: bool = False,
        variant: str = "scf",
    ):
        super().__init__(inst, activations, linear_relax)
        assert variant in [
            "scf",
            "scf_cuts_2",
            "scf_cuts_3",
            "scf_sep_cuts",
            "scf_start",
            "scf_cuts_2_start",
        ], f"Invalid variant: {variant}"
        self.name = variant

        self.f = {}  # keys are (i,j) and values are the flow i->j
        self.x = {}  # keys are (i,j) and is 1 if edge (i,j) is used (has flow > 0)
        self.y = {}  # keys are (i) and is 1 if node i is visited
        self.z = None

        if "scf_cuts_" in self.name:
            self.activations["cutset"] = True
        else:
            self.activations["cutset"] = False

        if self.name == "scf_sep_cuts":
            self.callback = create_callback()
            self.solver.Params.lazyConstraints = 1

    def define_variables(self):
        var_type = gp.GRB.CONTINUOUS if self.linear_relax else gp.GRB.BINARY
        for i in self.instance.N:
            self.y[i] = self.solver.addVar(vtype=var_type, name=f"y_{i}", lb=0, ub=1)
        for i in self.instance.N_0:
            for j in self.instance.N_0:
                self.f[i, j] = self.solver.addVar(
                    vtype=gp.GRB.CONTINUOUS,
                    name=f"f_{i}_{j}",
                    lb=0,
                    ub=self.instance.T_max,
                )
                self.x[i, j] = self.solver.addVar(
                    vtype=var_type, name=f"x_{i}_{j}", lb=0, ub=1
                )
        self.z = self.solver.addVar(vtype=gp.GRB.CONTINUOUS, name="z", lb=0, ub=1)

        if self.name == "scf_sep_cuts":
            self.solver._x = self.x
            self.solver._y = self.y

        if "start" in self.name:
            gh = GreedyHeuristic(self.instance)
            start = gh.run_heuristics(time_limit=30, n_solutions=1)[0]
            self._set_initial_solution(start)

    def _set_initial_solution(self, start: Solution):
        y = {i: sum(start.y[i, k] for k in self.instance.K) for i in self.instance.N}
        x = {
            (i, j): sum(start.x[i, j, k] for k in self.instance.K)
            for i in self.instance.N_0
            for j in self.instance.N_0
        }
        f = {(i, j): 0 for i in self.instance.N_0 for j in self.instance.N_0}
        routes = {k: start.routes[k][0] for k in self.instance.K}
        for k, route in routes.items():
            route = route[::-1]
            total_t = 0
            for j, i in zip(route[:-1], route[1:]):
                assert x[i, j] == 1, f"Edge ({i}, {j}) is not used in any route"
                f[i, j] = self.instance.t[i, j] + total_t
                total_t += self.instance.t[i, j]

        # Set start solution
        for i in self.instance.N:
            self.y[i].Start = y[i]
        for i in self.instance.N_0:
            for j in self.instance.N_0:
                self.x[i, j].Start = x[i, j]
                self.f[i, j].Start = f[i, j]
        self.z.Start = start.obj

    def constraint_define_obj(self):
        for c in self.instance.C:
            self.solver.addConstr(
                self.z * self.instance.tau[c]
                <= gp.quicksum(
                    self.instance.alpha[i, c] * self.y[i] for i in self.instance.N
                ),
                name=f"define_obj_{c}",
            )

    def constraint_link_flow_and_edges(self):
        for i in self.instance.N_0:
            for j in self.instance.N_0:
                self.solver.addConstr(
                    self.f[i, j] <= self.instance.T_max * self.x[i, j],
                    name=f"link_flow_and_edges_1_{i}_{j}",
                )

    def constraint_leave_depot_route(
        self,
    ):  # K routes must leave the depot (hard constraint)
        self.solver.addConstr(
            gp.quicksum(self.x[0, i] for i in self.instance.N) == len(self.instance.K),
            name=f"leave_depot_route",
        )

    def constraint_enter_depot_route(self):  # K routes must enter the depot
        self.solver.addConstr(
            gp.quicksum(self.x[i, 0] for i in self.instance.N) == len(self.instance.K),
            name=f"enter_depot_route",
        )

    def constraint_enter_depot_flow(
        self,
    ):  # The flow entering the depot must be equal to the cost of the edge
        for i in self.instance.N:
            self.solver.addConstr(
                self.f[i, 0] == self.x[i, 0] * self.instance.t[i, 0],
                name=f"enter_depot_flow_{i}",
            )

    def constraint_flow_visit(self):
        for i in self.instance.N:
            self.solver.addConstr(
                gp.quicksum(self.f[j, i] - self.f[i, j] for j in self.instance.N_0)
                == gp.quicksum(
                    self.x[j, i] * self.instance.t[j, i] for j in self.instance.N_0
                ),
                name=f"flow_visit_{i}",
            )

    def constraint_route_visit(self):
        for i in self.instance.N:
            self.solver.addConstr(
                gp.quicksum(self.x[j, i] + self.x[i, j] for j in self.instance.N_0)
                == 2 * self.y[i],
                name=f"route_visit_{i}",
            )

    def constraint_route_visit_hard(self):
        for i in self.instance.N:
            self.solver.addConstr(
                gp.quicksum(self.x[j, i] for j in self.instance.N_0) == self.y[i],
                name=f"route_visit_hard_in_{i}",
            )
            self.solver.addConstr(
                gp.quicksum(self.x[i, j] for j in self.instance.N_0) == self.y[i],
                name=f"route_visit_hard_out_{i}",
            )

    def constraint_not_stay(self):
        for i in self.instance.N_0:
            self.solver.addConstr(self.x[i, i] == 0, name=f"not_stay_{i}")

    def constraint_cutset(self):
        # Use itertools to find all subsets of N
        if self.name == "scf_cuts_3":
            max_size = 3
        elif "scf_cuts_2" in self.name:
            max_size = 2
        else:
            raise ValueError(
                f"Applying cutset constraints to {self.name} is not allowed"
            )
        subsets = []
        for i in range(1, max_size + 1):
            subsets.extend(itertools.combinations(self.instance.N, i))
        for subset in subsets:
            for h in subset:
                delta = {
                    (i, j)
                    for (i, j) in self.x.keys()
                    if i in subset and j not in subset
                }
                self.solver.addConstr(
                    gp.quicksum(self.x[i, j] for (i, j) in delta) >= self.y[h],
                    name=f"cutset_{subset}_{h}",
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


def find_min_cut(x, y):
    g = nx.DiGraph()
    g.add_weighted_edges_from([(i, j, value) for (i, j), value in x.items()])
    N = set(g.nodes)
    for i in N - {0}:
        flow = nx.maximum_flow_value(g, 0, i, capacity="weight")
        if flow < y[i] - 0.5:
            val, partition = nx.minimum_cut(g, 0, i, capacity="weight")
            outgoing_edges = [
                (i, j) for i, j in g.edges if i in partition[0] and j in partition[1]
            ]
            return outgoing_edges, i


def add_cutset_constraint(model: gp.Model, where):
    if (
        where == gp.GRB.Callback.MIPNODE
        and model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) == gp.GRB.OPTIMAL
    ):
        x = model.cbGetNodeRel(model._x)
        y = model.cbGetNodeRel(model._y)
        add_cut_to_formulation(model, x, y)


def create_callback():
    def partial_func(model, where):
        add_cutset_constraint(model, where)

    return partial_func


def add_cut_to_formulation(model: gp.Model, x, y):
    res = find_min_cut(x, y)
    if res is None:
        return
    outgoing_edges, node = find_min_cut(x, y)
    model._num_lazy_constraints_added += 1
    model.cbLazy(
        gp.quicksum(model._x[i, j] for i, j in outgoing_edges) >= model._y[node]
    )
