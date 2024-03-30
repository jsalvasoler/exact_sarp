import math
import warnings

import networkx as nx

from src.formulations.cutset_formulation import CutSetFormulation
from src.formulations.mtz_formulation import MTZFormulation
from utils import Formulation, Instance, Solution
from formulations.scf_formulation import SCFFormulation
from formulations.mtz_opt_formulation import MTZOptFormulation
import random
import gurobipy as gp


def run_experiment_mtzopt_vs_scf(draw, seed=1):
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
        warnings.warn("SCF formulation is infeasible")
        return True

    for k, v in scf_form.y.items():
        if v.X > 0:
            print(f"y_{k}: {v.X}")
    for k, v in scf_form.x.items():
        if v.X > 0:
            print(f"x_{k}: {v.X}")
    for k, v in scf_form.f.items():
        if v.X > 0:
            print(f"f_{k}: {v.X}")
    for k, v in instance.t.items():
        print(f"t_{k}: {v}")

    x = {k: v.X for k, v in scf_form.x.items()}
    y = {k: v.X for k, v in scf_form.y.items()}
    f = {k: v.X for k, v in scf_form.f.items()}
    u = {
        i: sum(scf_form.x[j, i].X * instance.t[j, i] for j in instance.N_0)
        + instance.T_max
        - sum(scf_form.f[j, i].X for j in instance.N_0)
        for i in instance.N
    }
    u[0] = instance.T_max

    for k, v in u.items():
        print(f"u_{k}: {v}")

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
            to_check_left = (
                x[i, j] * instance.t[i, j]
                + 0
                - instance.T_max * sum(x[k, j] for k in instance.N_0 if k != i)
            )
            to_check_right = x[i, j] * instance.t[i, j] + instance.T_max * (x[i, j] - 1)
            to_check_left = round(to_check_left, 4)
            to_check_right = round(to_check_right, 4)
            print(
                f"{to_check_left} >= {to_check_right}: {True if to_check_left >= to_check_right else False} (i={i}, j={j})"
            )
            to_check = to_check_left >= to_check_right

            satisfied_mtz[i, j] = to_check
    satisfied_extra = {}
    for i in instance.N:
        print(f"Extra constraints for {i}")
        lb = x[0, i] * instance.t[0, i]
        satisfied_extra[i] = lb <= u[i] <= instance.T_max
        print(
            f"   {lb} <= {u[i]} <= {instance.T_max}: {True if satisfied_extra[i] else False}"
        )

    print(
        f"Satisfied MTZ constraints: {sum(satisfied_mtz.values())} / {len(satisfied_mtz)}"
    )
    print(
        f"Satisfied extra constraints: {sum(satisfied_extra.values())} / {len(satisfied_extra)}"
    )

    if sum(satisfied_mtz.values()) != len(satisfied_mtz) or sum(
        satisfied_extra.values()
    ) != len(satisfied_extra):
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
    G.add_edges_from(
        [(i, j) for i in instance.N_0 for j in instance.N_0 if i != j and x[i, j] > 0]
    )
    pos = nx.circular_layout(G)
    nx.draw_networkx(G, pos=pos, with_labels=True)
    node_labels = {i: f"y_{i}={y[i]}\nu_{i}={u[i]}" for i in instance.N if y[i] > 0}
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels)
    edge_labels = {
        (i, j): f"x_{i, j}={x[i, j]}\nf_{i, j}={x[i, j]}\nt_{i, j}={instance.t[i, j]}"
        for i in instance.N_0
        for j in instance.N_0
        if i != j and x[i, j] > 0
    }
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.show()


def infeasible_analysis(solver):
    solver.computeIIS()
    for c in solver.getConstrs():
        if c.IISConstr:
            print(f"\t{c.constrname}: {solver.getRow(c)} {c.Sense} {c.RHS}")
    for v in solver.getVars():
        if v.IISLB:
            print(f"\t{v.varname} >= {v.LB}")
        if v.IISUB:
            print(f"\t{v.varname} =< {v.UB}")
    raise Exception("Infeasible")


def finding_counter_example_mtzopt_vs_scf(seed=0):
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
        (2, 2): 0,
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
    print("MTZ solution:")
    for i in instance.N_0:
        for j in instance.N_0:
            if i != j:
                print(f"x_{i, j} = {mtz_opt.x[i, j].X}")
                print(f"t_{i, j} = {instance.t[i, j]}")
    for i in instance.N_0:
        if i != 0:
            print(f"y_{i} = {mtz_opt.y[i].X}")
        print(f"u_{i} = {mtz_opt.u[i].X}")
    print("-----------------")

    # Print SCF solution:
    print("SCF solution:")
    for i in instance.N_0:
        for j in instance.N_0:
            if i != j:
                print(f"x_{i, j} = {scf.x[i, j].X}")
                print(f"f_{i, j} = {scf.f[i, j].X}")
                print(f"t_{i, j} = {instance.t[i, j]}")
    for i in instance.N:
        print(f"y_{i} = {scf.y[i].X}")

    print("-----------------")
    print(f"alpha: {instance.alpha}")
    print(f"MTZ: {mtz_relax}")
    print(f"SCF: {scf_relax}")


def experiments_mtzopt_vs_scf():
    for seed in range(1, 2):
        print(f"Seed: {seed}")
        if not run_experiment_mtzopt_vs_scf(draw=False, seed=seed):
            print("False!")
            break

    print("Done")


def run_experiment_mtz_vs_scf(seed, results):
    N = random.randint(2, 6)
    K = random.randint(1, 2)
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
        (2, 2): 0,
    }
    instance = Instance(N, 1, 1, 1, t=t, seed=seed)
    instance.print()

    scf = SCFFormulation(instance, linear_relax=True)
    scf.formulate()
    scf.solver.optimize()
    mtz = MTZFormulation(instance, linear_relax=True)
    mtz.formulate()
    mtz.solver.optimize()

    # Print both solutions
    print("SCF solution:")
    for i in instance.N_0:
        for j in instance.N_0:
            if i != j:
                print(f"x_{i, j} = {scf.x[i, j].X}")
                print(f"f_{i, j} = {scf.f[i, j].X}")
                print(f"t_{i, j} = {instance.t[i, j]}")
    for i in instance.N:
        print(f"y_{i} = {scf.y[i].X}")

    print("-----------------")
    print("MTZ solution:")
    for i in instance.N_0:
        for j in instance.N_0:
            if i != j:
                print(f"x_{i, j} = {mtz.x[i, j, 1].X}")
                print(f"t_{i, j} = {instance.t[i, j]}")
    for i in instance.N:
        if i != 0:
            print(f"y_{i} = {mtz.y[i, 1].X}")
        print(f"u_{i} = {mtz.u[i].X}")
    print("-----------------")

    try:
        scf_relax = scf.solver.objVal
        mtz_relax = mtz.solver.objVal
        if abs(scf_relax - mtz_relax) < 1e-6:
            results["equal"] += 1
        elif scf_relax > mtz_relax:
            results["scf_higher"] += 1
            results["example_scf_higher"].append((seed, scf_relax, mtz_relax))
        else:
            results["mtz_higher"] += 1
            results["example_mtz_higher"].append((seed, scf_relax, mtz_relax))
    except Exception as e:
        results["infeasible"] += 1
    return results


def run_experiment_mtz_vs_scf_check_translation(seed):
    print(f"-" * 40)
    print(f"-" * 40)
    instance = Instance(6, 3, 8, 2, seed=seed)
    instance.print()
    scf = SCFFormulation(instance, linear_relax=True)
    scf.formulate()
    scf.solver.optimize()
    try:
        x = {k: v.X for k, v in scf.x.items()}
        y = {k: v.X for k, v in scf.y.items()}
        f = {k: v.X for k, v in scf.f.items()}
    except Exception as e:
        return
    print("x:", {k: v for k, v in x.items() if v > 0})
    print("y:", {k: v for k, v in y.items() if v > 0})
    print("f:", {k: v for k, v in f.items() if v > 0})

    K = list(range(1, len(instance.K) + 1))

    x_mtz = {(i, j, k): x[i, j] / len(instance.K) for (i, j) in x.keys() for k in K}
    y_mtz = {
        (i, k): sum(x_mtz[i, j, k] for j in instance.N_0)
        for i in instance.N_0
        for k in K
    }

    def F(u):
        assert len(u) <= len(instance.N)
        t = [(u[i], i) for i in u]
        t.sort()
        return {i: j + 1 for j, (_, i) in enumerate(t)}

    u_times = {
        i: sum(x[j, i] * instance.t[j, i] for j in instance.N_0)
        + instance.T_max
        - sum(f[j, i] for j in instance.N_0)
        for i in instance.N
    }
    u_mtz = {}
    u_order = F(u_times)
    for i in instance.N:
        u_mtz[i] = u_order[i]

    print("x_mtz:", {k: v for k, v in x_mtz.items() if v > 0})
    print("y_mtz:", {k: v for k, v in y_mtz.items() if v > 0})
    print("u_mtz:", {k: v for k, v in u_mtz.items()})
    print("*u_times:", {k: v for k, v in u_times.items()})

    # Check MTZ constraints
    return check_mtz_constraints(instance, x_mtz, y_mtz, u_mtz, f=f, x=x)


def experiments_mtz_vs_scf():
    try_to_see_stronger_or_weaker_mtz_vs_scf()
    # N = 1000
    # for seed in range(N):
    #     seed = -seed
    #     print(f'Running experiment {seed} of {N}')
    #     run_experiment_mtz_vs_scf_check_translation(seed)


def check_mtz_constraints(instance, x_mtz, y_mtz, u_mtz, f=None, x=None):
    K = list(range(1, len(instance.K) + 1))
    for i in instance.N_0:
        for k in K:
            assert (
                sum(x_mtz[i, j, k] for j in instance.N_0) == y_mtz[i, k]
            ), f"sum of x over j (1) failed for i={i}, k={k}"
            assert (
                abs(sum(x_mtz[j, i, k] for j in instance.N_0) - y_mtz[i, k]) < 1e6
            ), f"sum of x over j (2) failed for i={i}, k={k}"
            assert x_mtz[i, i, k] == 0, f"x[i, i] = 0 failed for i={i}"
            assert (
                0 <= round(y_mtz[i, k], 7) <= 1
            ), f"y_mtz[i, k] in [0, 1] failed for i={i}, k={k}"
        if i != 0:
            assert (
                round(sum(y_mtz[i, k] for k in K), 7) <= 1
            ), f"sum of y over k failed for i={i}\n{sum(y_mtz[i, k] for k in K)}"
    for k in K:
        lhs = sum(
            instance.t[i, j] * x_mtz[i, j, k]
            for i in instance.N_0
            for j in instance.N_0
        )
        rhs = instance.T_max
        assert round(lhs, 7) <= round(
            rhs, 7
        ), f"sum of t*x over i,j failed for k={k}\n{lhs} = lhs <= rhs = {rhs}"
    for k in K:
        for i in instance.N:
            for j in instance.N:
                if i == j:
                    continue
                lhs = u_mtz[i] - u_mtz[j] + len(instance.N) * x_mtz[i, j, k]
                rhs = len(instance.N)
                assert (
                    lhs <= rhs
                ), f"MTZ failed for i={i}, j={j}, k={k}\n{lhs} = lhs <= rhs = {rhs}"
    return True


def try_to_see_stronger_or_weaker_mtz_vs_scf():
    results = {
        "scf_higher": 0,
        "mtz_higher": 0,
        "equal": 0,
        "infeasible": 0,
        "example_scf_higher": [],
        "example_mtz_higher": [],
    }
    N_exec = 2000
    for seed in range(0, N_exec):
        seed = -seed
        if seed != -1685:
            continue
        print(f"Seed: {seed}")
        results = run_experiment_mtz_vs_scf(seed, results)
    print(results)


def experiments_scf_vs_cutset():
    try_to_see_stronger_or_weaker("mtz", "cutset")


def try_to_see_stronger_or_weaker(form_1, form_2):
    results = {
        f"{form_1}_higher": 0,
        f"{form_2}_higher": 0,
        "equal": 0,
        "infeasible": 0,
        f"example_{form_1}_higher": [],
        f"example_{form_2}_higher": [],
    }
    form_1 = formulations[form_1]
    form_2 = formulations[form_2]
    N_exec = 2500
    for seed in range(0, N_exec):
        seed = -seed
        if seed != -14:
            continue
        print(f"Seed: {seed}")
        results = run_experiment_stronger_weaker(seed, results, form_1, form_2)
    print(results)


def run_experiment_stronger_weaker(seed, results, form_1, form_2):
    random.seed(seed)
    N = random.randint(2, 8)
    K = random.randint(1, min(N, 4))
    t = {
        (0, 1): 1 / 2,
        (1, 0): 1 / 2,
        (0, 2): 1,
        (2, 0): 1,
        (1, 2): 1,
        (2, 1): 1,
        (0, 0): 0,
        (1, 1): 0,
        (2, 2): 0,
    }
    alpha = {(1, 1): 0, (2, 1): 1}
    instance = Instance(N, K, random.randint(10, 40), 1, t=None, alpha=None, seed=seed)
    instance.print()

    scf = form_1(instance, linear_relax=True)
    scf.formulate()
    scf.solver.optimize()
    if type(form_2) is CutSetFormulation:
        cutset = form_2(instance, linear_relax=True, full_model=True)
    else:
        cutset = form_2(instance, linear_relax=True)
    cutset.formulate()
    cutset.solver.optimize()

    try:
        scf_relax = scf.solver.objVal
        cutset_relax = cutset.solver.objVal
        if abs(scf_relax - cutset_relax) < 1e-6:
            results["equal"] += 1
        elif scf_relax > cutset_relax:
            results[f"{scf.name}_higher"] += 1
            results[f"example_{scf.name}_higher"].append(
                (seed, scf_relax, cutset_relax, len(instance.N))
            )
        else:
            results[f"{cutset.name}_higher"] += 1
            results[f"example_{cutset.name}_higher"].append(
                (seed, scf_relax, cutset_relax, len(instance.N))
            )
    except Exception as e:
        results["infeasible"] += 1
    return results


formulations = {
    "mtz": MTZFormulation,
    "cutset": CutSetFormulation,
    "scf": SCFFormulation,
    "mtz_opt": MTZOptFormulation,
}


if __name__ == "__main__":
    # experiments_mtzopt_vs_scf()
    # finding_counter_example_mtzopt_vs_scf()
    # experiments_mtz_vs_scf()
    experiments_scf_vs_cutset()
