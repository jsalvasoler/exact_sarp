from optimizer import Optimizer
from config import Config
from formulations.cutset_formulation import CutSetFormulation
from formulations.mtz_formulation import MTZFormulation
from formulations.mtz_opt_formulation import MTZOptFormulation
from formulations.scf_formulation import SCFFormulation
from instance_loader import InstanceLoader
from heuristics.greedy_heuristic import GreedyHeuristic
from heuristics.rl_heuristic import RLHeuristic
from heuristics.grasp import GRASP

import time
import pandas as pd
from pyinstrument import Profiler

formulations = {
    "mtz": MTZFormulation,
    "cutset": CutSetFormulation,
    "scf": SCFFormulation,
    "mtz_opt": MTZOptFormulation,
}


def main():
    config = Config()
    instance_loader = InstanceLoader(config)
    instances = instance_loader.load_instances(id_indices=False)

    for i, (name, instance) in enumerate(instances.items()):
        print(
            f"\nInstance {i + 1}/{len(instances)}: \n  id = {name[:2]}, name = {name[3:-4]}\n"
        )
        print(f" -- Instance information: {instance.instance_results}\n\n")
        instance.print()

        formulation = define_formulation(config, instance)
        optimizer = Optimizer(formulation, config)
        optimizer.run()


def define_formulation(config, instance):
    if "scf" in config.formulation:
        formulation = formulations.get("scf")(
            instance,
            config.activations.get(config.formulation, {}),
            variant=config.formulation,
        )
    else:
        formulation = formulations.get(config.formulation)(
            instance, config.activations.get(config.formulation, {})
        )
    return formulation


def big_experiment():
    config = Config()
    config.results_file = "big_results.csv"

    assert (
        config.n_instances_main is None and config.n_instances_big
    ), "In big experiment, instance number is provided by n_instances_big"
    assert config.time_limit == 60, "Big experiment needs time_limit to be 60"

    results = pd.read_csv(config.results_filepath, sep=";", decimal=",")

    # Find instances + formulations that have already been solved
    # That means either the solve_time is greater than the time_limit or the mip_gap is zero + tolerance
    solved = results.loc[
        (results["type"] == "case")
        & (
            (results["solve_time"] >= config.time_limit * 60)
            | (results["mip_gap"].abs() < 1e-6)
        ),
        ["id", "formulation"],
    ].values
    ids = list(range(1, 24))
    # ids = [60, 66, 77, 79, 82, 84]  # Fast ids
    form_names = ["mtz_opt"]

    all_executions = {
        (instance_id, form_name) for instance_id in ids for form_name in form_names
    }
    # to_execute = sorted(list(all_executions - set(map(tuple, solved))), key=lambda x: x[0])
    to_execute = sorted(list(all_executions), key=lambda x: x[0])
    print(f"All executions: {sorted(all_executions)}")
    print(f"To execute: {to_execute}")

    instance_loader = InstanceLoader(config)
    instances = instance_loader.load_instances()

    for i, (instance_id, formulation_name) in enumerate(
        to_execute[: config.n_instances_big]
    ):
        config.set_formulation(formulation_name)
        instance = instances[instance_id]
        print(
            f"\nInstance {i + 1}/{config.n_instances_big}: \n  "
            f"id = {instance_id}, name = {instance.name}, formulation = {formulation_name}\n"
            f"instance information: {instance.instance_results}\n\n"
        )

        formulation = define_formulation(config, instance)
        optimizer = Optimizer(formulation, config)
        optimizer.run()


def instance_difficulty_experiment():
    instance_name, instance_id = "large_RC50_K4T5", 42
    config = Config()
    assert (
        config.instance_type == "large" and config.formulation == "scf"
    ), "Check config settings. Need large + scf"
    config.results_large = "results_instance_diff.csv"

    instance_loader = InstanceLoader(config)
    instances = instance_loader.load_instances()

    instance = instances[instance_id]

    T_max_set = [1, 2, 3, 4, 5, 6, 7]
    K_size_set = [6]
    to_execute = [(T_max, K_size) for T_max in T_max_set for K_size in K_size_set]
    for i, (T_max, K_size) in enumerate(to_execute):
        instance.update_T_max(T_max)
        instance.update_K_size(K_size)

        print(
            f"\nInstance {i + 1}/{len(to_execute)}: \n  T_max = {T_max}, K_size = {K_size}\n\n"
        )
        instance.print()

        formulation = formulations.get(config.formulation)(
            instance, config.activations.get(config.formulation, {})
        )
        optimizer = Optimizer(formulation, config)
        optimizer.run()

    return None


def test_heuristic_methods():

    config = Config()
    # assert config.instance_type == "large", "Check config settings. Need type = large"
    instance_loader = InstanceLoader(config)
    instances = instance_loader.load_instances(id_indices=False)
    results = {}

    try:
        results_df = pd.read_csv("results/heuristic_results.csv", sep=";", decimal=",")
    except FileNotFoundError:
        results_df = pd.DataFrame(
            columns=["instance", "method", "objective", "runtime"]
        )

    for instance_name, instance in list(instances.items()):
        print(f"\nInstance: {instance_name}")

        Z_TS = instance.instance_results["Z_TS"]
        Z_GH = Z_TS / (1 + instance.instance_results["ts_vs_gh_gap"] / 100)
        greedy_heuristic = GreedyHeuristic(instance)
        rl_heuristic = RLHeuristic(instance, n_episodes=10000)
        grasp = GRASP(instance, n_trials=20)

        # results[instance_name] = (
        #     Z_TS,
        #     Z_GH,
        #     solution.obj,
        #     det_sol.obj,
        #     rl_solution.obj,
        #     grasp_solution.obj,
        # )

        rows_to_add = []

        if results_df[
            (results_df["instance"] == instance_name)
            & (results_df["method"] == "greedy_heuristic_det")
        ].empty:
            start_time = time.time()
            det_sol = greedy_heuristic.build_solution(0)
            runtime_det = time.time() - start_time
            row = [instance_name, "greedy_heuristic_det", det_sol.obj, runtime_det]
            rows_to_add.append(row)
        if results_df[
            (results_df["instance"] == instance_name)
            & (results_df["method"] == "TS_paper")
        ].empty:
            row = [instance_name, "TS_paper", Z_TS, 0]
            rows_to_add.append(row)
        if results_df[
            (results_df["instance"] == instance_name)
            & (results_df["method"] == "GH")
        ].empty:
            row = [instance_name, "GH", Z_GH, 0]
            rows_to_add.append(row)
        if results_df[
            (results_df["instance"] == instance_name)
            & (results_df["method"] == "greedy_heuristic")
        ].empty:
            start_time = time.time()
            solutions = greedy_heuristic.run_heuristics(30, 50)
            solution = solutions[0]
            runtime_gh = time.time() - start_time
            row = [instance_name, "greedy_heuristic", solution.obj, runtime_gh]
            rows_to_add.append(row)

            grasp_solution = grasp.run_grasp_after_sampling(30, solution_pool=solutions)
            runtime_grasp = time.time() - start_time
            row = [instance_name, "grasp", grasp_solution.obj, runtime_grasp]
            rows_to_add.append(row)

        if results_df[
            (results_df["instance"] == instance_name)
            & (results_df["method"] == "rl_heuristic")
        ].empty:
            start_time = time.time()
            rl_solution = rl_heuristic.run()
            runtime_rl = time.time() - start_time
            row = [instance_name, "rl_heuristic", rl_solution.obj, runtime_rl]
            rows_to_add.append(row)

        for row in rows_to_add:
            results_df.loc[len(results_df), :] = row

        results_df.to_csv(
            "results/heuristic_results.csv", sep=";", decimal=",", index=False
        )

    # n_better, n_worse, beat_TS, random_wins_det = 0, 0, 0, 0
    # gap, gap_ts, gap_rl_ts, gap_rl_gh = 0, 0, 0, 0
    # for instance_name, (Z_TS, Z_GH, obj, det_obj, rl_obj) in results.items():
    #     gap += (obj - Z_GH) / Z_GH if Z_GH > 0 else 0
    #     gap_ts += (obj - Z_TS) / Z_TS if Z_TS > 0 else 0
    #     gap_rl_ts += (rl_obj - Z_TS) / Z_TS if Z_TS > 0 else 0
    #     gap_rl_gh += (rl_obj - Z_GH) / Z_GH if Z_GH > 0 else 0
    #     if Z_GH > obj:
    #         n_worse += 1
    #     else:
    #         n_better += 1
    #     if Z_TS <= obj:
    #         beat_TS += 1
    #     if obj > det_obj:
    #         random_wins_det += 1
    # print("----------- Results --------------")
    # print(f"Z_GH > Z_gh: {n_worse} / {len(results)} times")
    # print(f"Z_TS <= Z_gh: {beat_TS} / {len(results)} times")
    # print(f"Z_gh > Z_det: {random_wins_det} / {len(results)} times")
    # print(f"AVG gap (Z_gh - Z_GH) / Z_GH %: {round(100 * gap / len(results), 3)}")
    # print(f"AVG gap (Z_gh - Z_TS) / Z_TS %: {round(100 * gap_ts / len(results), 3)}")
    # print(f"AVG gap (Z_rl - Z_GH) / Z_GH %: {round(100 * gap_rl_gh / len(results), 3)}")
    # print(f"AVG gap (Z_rl - Z_TS) / Z_TS %: {round(100 * gap_rl_ts / len(results), 3)}")


if __name__ == "__main__":
    profiler = Profiler()
    profiler.start()
    # main()
    # big_experiment()
    # instance_difficulty_experiment()
    test_heuristic_methods()
    # theory_main()
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
