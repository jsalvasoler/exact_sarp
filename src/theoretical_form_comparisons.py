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

    instance = Instance(10, 3, 10, 1, seed=seed)
    scf_form = SCFFormulation(instance, linear_relax=True)
    scf_form.formulate()

    profits = {i: random.randint(-3, 10) for i in instance.N}
    # scf_form.solver.setObjective(gp.quicksum(profits[i] * scf_form.y[i] for i in instance.N), gp.GRB.MAXIMIZE)
    scf_form.solver.optimize()

    if scf_form.solver.status == gp.GRB.INFEASIBLE:
        warnings.warn('SCF formulation is infeasible')
        return True

    for k, v in scf_form.y.items():
        if v.X > 0:
            print(f'y_{k}: {v.X}')
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
        lb = x[0, i] * instance.t[0, i]
        satisfied_extra[i] = lb <= u[i] <= instance.T_max
        print(f'   {lb} <= {u[i]} <= {instance.T_max}: {True if satisfied_extra[i] else False}')

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
    t = {
        (0, 1): 1 / 2,
        (1, 0): 1 / 2,
        (0, 2): 1,
        (2, 0): 1,
        (1, 2): 1,
        (2, 1): 1,
        (0, 0): 0,
        (1, 1): 0,
        (2, 2): 0
    }
    instance = Instance(N, 1, 1, 1, t=t, seed=seed)
    instance.print()
    mtz_opt = MTZOptFormulation(instance, linear_relax=True)
    mtz_opt.formulate()
    mtz_opt.solver.optimize()
    mtz_relax = mtz_opt.solver.objVal

    scf = SCFFormulation(instance, linear_relax=True)
    scf.formulate()
    scf.solver.optimize()
    scf_relax = scf.solver.objVal

    # Print MTZ solution:
    print('MTZ solution:')
    for i in instance.N_0:
        for j in instance.N_0:
            if i != j:
                print(f'x_{i, j} = {mtz_opt.x[i, j].X}')
                print(f't_{i, j} = {instance.t[i, j]}')
    for i in instance.N_0:
        if i != 0:
            print(f'y_{i} = {mtz_opt.y[i].X}')
        print(f'u_{i} = {mtz_opt.u[i].X}')
    print('-----------------')

    # Print SCF solution:
    print('SCF solution:')
    for i in instance.N_0:
        for j in instance.N_0:
            if i != j:
                print(f'x_{i, j} = {scf.x[i, j].X}')
                print(f'f_{i, j} = {scf.f[i, j].X}')
                print(f't_{i, j} = {instance.t[i, j]}')
    for i in instance.N:
        print(f'y_{i} = {scf.y[i].X}')

    print('-----------------')
    print(f'alpha: {instance.alpha}')
    print(f'MTZ: {mtz_relax}')
    print(f'SCF: {scf_relax}')


if __name__ == '__main__':
    # for seed in range(1, 2):
    #     print(f'Seed: {seed}')
    #     if not run_experiment(draw=False, seed=seed):
    #         print('False!')
    #         break
    #
    # print('Done')
    finding_counter_example()
