import warnings

from utils import Formulation, Instance, Solution
from formulations.scf_formulation import SCFFormulation
from formulations.mtz_opt_formulation import MTZOptFormulation
import random
import gurobipy as gp


def run_experiment(draw, seed=1):
    """
    This experiment checks whether it is true or not the P_SCF \subset P_MTZOPT by doing the following:
    1. Generate a random instance
    2. Solve the LR of the SCF formulation
    3. Build a solution for the MTZOPT formulation using the solution of the LR of the SCF formulation. Define the
    u_i values based on the f_ij values of the SCF formulation.
    4. Check whether the solution is feasible for the LR of the MTZOPT formulation
    """

    instance = Instance(10, 3, 8, 1, seed=seed)
    scf_form = SCFFormulation(instance, linear_relax=True)
    scf_form.formulate()

    profits = {i: random.randint(-3, 10) for i in instance.N}
    scf_form.solver.setObjective(gp.quicksum(profits[i] * scf_form.y[i] for i in instance.N), gp.GRB.MAXIMIZE)
    scf_form.solver.optimize()

    if scf_form.solver.status == gp.GRB.INFEASIBLE:
        warnings.warn('SCF formulation is infeasible')
        return True

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
    f = {k: v.X for k, v in scf_form.f.items()}
    u = {i: sum(scf_form.x[j, i].X * instance.t[j, i] for j in instance.N_0) + instance.T_max - sum(
        scf_form.f[j, i].X for j in instance.N_0) for i in instance.N}
    u[0] = instance.T_max

    for k, v in u.items():
        print(f'u_{k}: {v}')

    if draw:
        draw_solution(instance, u, x, y)

    # Check whether the solution is feasible for the MTZOPT formulation
    satisfied_mtz = {}
    for i in instance.N:
        for j in instance.N:
            if i == j:
                continue
            # to_check_left = sum(x[k, j] * instance.t[k, j] for k in instance.N_0) + sum(f[i, k] for k in instance.N_0)
            # to_check_left = sum(x[k, j] * instance.t[k, j] for k in instance.N_0) - \
            # sum(x[k, i] * instance.t[k, i] for k in instance.N_0) + \
            to_check_left = x[i, j] * instance.t[i, j] + \
                            0 - \
                            instance.T_max * sum(x[k, j] for k in instance.N_0 if k != i)
            to_check_right = x[i, j] * instance.t[i, j] + instance.T_max * (x[i, j] - 1)
            to_check_left = round(to_check_left, 4)
            to_check_right = round(to_check_right, 4)
            print(f'{to_check_left} >= {to_check_right}: {True if to_check_left >= to_check_right else False} (i={i}, j={j})')
            to_check = to_check_left >= to_check_right

            satisfied_mtz[i, j] = to_check
    satisfied_extra = {}
    for i in instance.N:
        print(f'Extra constraints for {i}')
        satisfied_extra[i] = 0 <= u[i] <= instance.T_max
        print(f'   {0} <= {u[i]} <= {instance.T_max}: {True if satisfied_extra[i] else False}')

    print(f'Satisfied MTZ constraints: {sum(satisfied_mtz.values())} / {len(satisfied_mtz)}')
    print(f'Satisfied extra constraints: {sum(satisfied_extra.values())} / {len(satisfied_extra)}')

    if sum(satisfied_mtz.values()) != len(satisfied_mtz) or sum(satisfied_extra.values()) != len(satisfied_extra):
        return False
    return True

    # print(f'Objective: {scf_form.solver.objVal}')
    # mtz_form = MTZOptFormulation(instance, linear_relax=True)
    # mtz_form.formulate()
    # mtz_form.impose_solution(x, y, u)
    # mtz_form.solver.setObjective(gp.quicksum(profits[i] * mtz_form.y[i] for i in instance.N), gp.GRB.MAXIMIZE)
    # mtz_form.solver.optimize()
    #
    # if mtz_form.solver.status == gp.GRB.INFEASIBLE:
    #     infeasible_analysis(mtz_form.solver)
    #     raise Exception('MTZ formulation is infeasible')
    #
    # assert abs(mtz_form.solver.ObjVal - scf_form.solver.ObjVal) < 1e-6


def draw_solution(instance, u, x, y):
    # Draw the graph with x, f values on the edges and y, u values on the nodes
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.DiGraph()
    G.add_nodes_from(instance.N_0)
    G.add_edges_from([(i, j) for i in instance.N_0 for j in instance.N_0 if i != j and x[i, j] > 0])
    pos = nx.circular_layout(G)
    nx.draw_networkx(G, pos=pos, with_labels=True)
    node_labels = {i: f'y_{i}={y[i]}\nu_{i}={u[i]}' for i in instance.N if y[i] > 0}
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels)
    edge_labels = {(i, j): f'x_{i, j}={x[i, j]}\nf_{i, j}={x[i, j]}\nt_{i, j}={instance.t[i, j]}' for i in
                   instance.N_0
                   for j in instance.N_0 if i != j and x[i, j] > 0}
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.show()


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


def finding_counter_example(seed=0):
    N = 2
    instance = Instance(N, 1, 10, 1, seed=seed)
    mtz_opt = MTZOptFormulation(instance, linear_relax=True)
    mtz_opt.formulate()

    scf = SCFFormulation(instance, activations={'flow_visit': False}, linear_relax=True)
    scf.solver = mtz_opt.solver
    scf.formulate()

    mtz_opt.solver = scf.solver
    mtz_opt.f = scf.f
    mtz_opt.solver.setObjective(
        gp.quicksum(mtz_opt.f[k, 1] + mtz_opt.f[1, k] - mtz_opt.x[k, 1] * instance.t[k, 1] for k in instance.N_0),
        # 1
    )

    mtz_opt.solver.optimize()
    print(f'Solution:')
    for v in mtz_opt.solver.getVars():
        print(f'\t{v.varName}: {v.x}')

    # Print constraints
    print(f'Constraints:')
    for c in mtz_opt.solver.getConstrs():
        print(f'\t{c.constrName}: {c.RHS} {c.sense} {mtz_opt.solver.getRow(c)}')

    x = {k: v.X for k, v in mtz_opt.x.items()}
    y = {k: v.X for k, v in mtz_opt.y.items()}
    u = {k: v.X for k, v in mtz_opt.u.items()}

    mtz_opt.impose_solution(x, y, u)
    mtz_opt.solver.optimize()

    if mtz_opt.solver.status == gp.GRB.INFEASIBLE:
        print('We found such example')
        return False
    return True

if __name__ == '__main__':
    # for seed in range(1, 100):
    #     print(f'Seed: {seed}')
    #     if not run_experiment(draw=False, seed=seed):
    #         print('False!')
    #         break
    #
    # print('Done')
    finding_counter_example()
