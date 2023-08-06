import warnings

from utils import Formulation, Instance, Solution
from formulations.scf_formulation import SCFFormulation
from formulations.mtz_opt_formulation import MTZOptFormulation
import random
import gurobipy as gp


def run_experiment(draw, seed=1):
    instance = Instance(10, 3, 8, 1, seed=seed)
    scf_form = SCFFormulation(instance, linear_relax=True)
    scf_form.formulate()

    profits = {i: random.randint(-3, 10) for i in instance.N}
    scf_form.solver.setObjective(gp.quicksum(profits[i] * scf_form.y[i] for i in instance.N), gp.GRB.MAXIMIZE)
    scf_form.solver.optimize()

    if scf_form.solver.status == gp.GRB.INFEASIBLE:
        warnings.warn('SCF formulation is infeasible')
        return

    for k, v in scf_form.y.items():
        if v.X > 0:
            print(f'y_{k}: {v.X}')
        print(f'{k}: {v.X}')
    for k, v in scf_form.x.items():
        if v.X > 0:
            print(f'x_{k}: {v.X}')
    for k, v in scf_form.f.items():
        if v.X > 0:
            print(f'f_{k}: {v.X}')
    for k, v in instance.t.items():
        print(f't_{k}: {v}')

    x = {k: v.X for k, v in scf_form.x.items()}
    y = {k: v.X for k, v in scf_form.y.items()}
    u = {i: sum(scf_form.x[j, i].X * instance.t[j, i] for j in instance.N_0) + instance.T_max - sum(
        scf_form.f[j, i].X for j in instance.N_0) for i in instance.N}
    u[0] = instance.T_max

    for k, v in u.items():
        print(f'u_{k}: {v}')

    if draw:
        # Draw the K5 graph with x, f values on the edges and y, u values on the nodes
        import networkx as nx
        import matplotlib.pyplot as plt
        G = nx.DiGraph()
        G.add_nodes_from(instance.N_0)
        G.add_edges_from([(i, j) for i in instance.N_0 for j in instance.N_0 if i != j and x[i, j] > 0])
        pos = nx.circular_layout(G)
        nx.draw_networkx(G, pos=pos, with_labels=True)
        node_labels = {i: f'y_{i}={y[i]}\nu_{i}={u[i]}' for i in instance.N if y[i] > 0}
        nx.draw_networkx_labels(G, pos=pos, labels=node_labels)
        edge_labels = {(i, j): f'x_{i, j}={x[i, j]}\nf_{i, j}={x[i, j]}\nt_{i, j}={instance.t[i, j]}' for i in instance.N_0
                       for j in instance.N_0 if i != j and x[i, j] > 0}
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
        plt.show()

    print(f'Objective: {scf_form.solver.objVal}')
    mtz_form = MTZOptFormulation(instance, linear_relax=True)
    mtz_form.formulate()
    mtz_form.impose_solution(x, y, u)
    mtz_form.solver.setObjective(gp.quicksum(profits[i] * mtz_form.y[i] for i in instance.N), gp.GRB.MAXIMIZE)
    mtz_form.solver.optimize()

    if mtz_form.solver.status == gp.GRB.INFEASIBLE:
        infeasible_analysis(mtz_form.solver)
        raise Exception('MTZ formulation is infeasible')

    assert abs(mtz_form.solver.ObjVal - scf_form.solver.ObjVal) < 1e-6


def infeasible_analysis(solver):
    solver.computeIIS()
    for c in solver.getConstrs():
        if c.IISConstr:
            print(f'\t{c.constrname}: {solver.getRow(c)} {c.Sense} {c.RHS}')
    for v in solver.getVars():
        if v.IISLB:
            print(f'\t{v.varname} >= {v.LB}')
        if v.IISUB:
            print(f'\t{v.varname} =< {v.UB}')
    raise Exception('Infeasible')


if __name__ == '__main__':
    for seed in range(1, 1000):
        print(f'Seed: {seed}')
        run_experiment(draw=False, seed=seed)
