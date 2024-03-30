import random
import time
from typing import Tuple, List

import numpy as np

from instance_loader import InstanceLoader
from utils import Instance, Solution
from config import Config


class GreedyHeuristic:
    """
    This class generates feasible solutions using a Greedy-based heuristic. The heuristic is based on the following
    rules:
    - Find K most critical characteristics according to average intra-characteristic distance
    - Start K routes that visit the closest nodes containing these characteristics
    - Iterate until impossible:
        - Take characteristic with the lowest coverage ratio
        - Find the node with this characteristic that is closest to an end of a partial route, and add it.
    Some randomness is added to the heuristic on the 'sorting' steps, in order to step away from the greedy policy. The
    randomness is added in the form of randomness * random.uniform(-1, 1) to the sorting keys of the orderings.

    Args:
        instance: Instance to find feasible solutions for.
        config: Config object.
        seed: Random seed.
    """

    def __init__(self, instance: Instance, seed: int = 42):
        random.seed(seed)
        self._instance = instance

        # By default, only the third selection step contains randomness
        self._active_selection_randomness = {
            "sel_1": 0,
            "sel_2": 0,
            "sel_3": 1,
            "sel_4": 0,
        }

        self._average_distance_matrix = self._compute_average_distance_matrix()
        self.randomness = 0

    def _compute_average_distance_matrix(self):
        """
        Compute the average distance matrix. For each critical characteristic c \in C, we compute the average distance
        between all nodes that carry the characteristic.

        Returns:
            dict: {characteristic: double}. Average distance matrix.
        """
        nodes_by_characteristic = {
            c: {i for i in self._instance.N if self._instance.alpha[i, c]}
            for c in self._instance.C
        }

        return {
            c: self._average_distance(nodes)
            for c, nodes in nodes_by_characteristic.items()
        }

    def _average_distance(self, nodes):
        """
        Compute the average distance between all nodes in the set nodes.
        Args:
            nodes: set of nodes.

        Returns:
            double: Average distance between all nodes in the set nodes.
        """
        pairs = {(i, j) for i in nodes for j in nodes if i < j}
        return sum(self._instance.t[i, j] for i, j in pairs) / len(pairs)

    def _coverage_ratios(self, visited):
        """
        Compute the coverage ratios of all characteristics in the instance, given the set of visited nodes.

        Args:
            visited: Set of visited nodes.

        Returns:
            dict: {characteristic: double}. Coverage ratios.
        """
        return {
            c: sum(self._instance.alpha[i, c] for i in self._instance.N if i in visited)
            / self._instance.tau[c]
            for c in self._instance.C
        }

    def build_solution(self, randomness: float = None, seed: int = None) -> Solution:
        """
        Build a solution using the greedy heuristic.

        Args:
            randomness: Randomness factor. If 0, the solution is deterministic. If > 0, the solution is random. The
            factor is added in the form of 1 * random.uniform(-randomness, randomness) to the sorting keys of the
            orderings.
            seed: Random seed.

        Returns:
            Solution: Solution built.
        """
        if seed:
            random.seed(seed)
        if randomness:
            self.randomness = randomness
        routes = {k: [0] for k in self._instance.K}
        route_durations = {k: 0 for k in self._instance.K}
        visited = {0}

        # SELECTION 1
        # Select the K characteristics with the lowest average distance, as computed in the average distance matrix.
        sort_key = (
            lambda c: self._average_distance_matrix[c]
            + self._active_selection_randomness["sel_1"]
            * self.randomness
            * random.uniform(-1, 1)
            * self._average_distance_matrix[c]
        )
        start_characteristics = sorted(self._instance.C, key=sort_key)[
            : len(self._instance.K)
        ]

        # SELECTION 2
        # Start K routes from the depot, one for each selected characteristic.
        for k, c in zip(self._instance.K, start_characteristics):
            # Identify the closest node that carries characteristic c.
            sort_key = (
                lambda i: self._instance.t[0, i]
                + self._active_selection_randomness["sel_2"]
                * self.randomness
                * random.uniform(-1, 1)
                * self._average_distance_matrix[c]
            )
            closest_node = min(
                {
                    i
                    for i in self._instance.N
                    if self._instance.alpha[i, c] and i not in visited
                },
                key=sort_key,
            )
            if self._instance.t[0, closest_node] * 2 > self._instance.T_max:
                closest_node = min(
                    {i for i in self._instance.N if not i in visited}, key=sort_key
                )
            self.add_i_to_route_of_k(closest_node, k, route_durations, routes, visited)

        while self._iterative_improvement(routes, route_durations, visited):
            pass

        # Add the depot to the end of each route.
        for k in self._instance.K:
            self.add_i_to_route_of_k(0, k, route_durations, routes, visited)
        return self._define_solution_instance(routes)

    def _iterative_improvement(self, routes, route_durations, visited) -> bool:
        """
        Perform iterative improvement on the solution. The improvement is done by selecting the characteristic with
        the lowest coverage ratio, and inserting the closest node that carries this characteristic into the route
        where it would be the cheapest to insert it.

        Args:
            routes: partial routes for each vehicle.
            route_durations: current duration of each partial route.
            visited: set of visited nodes.

        Returns:
            bool: True if an improvement was done, False otherwise.
        """
        # Update coverage ratios
        coverage_ratios = self._coverage_ratios(visited)

        # SELECTION 3
        # Sort characteristics according to their coverage ratio
        min_cr, max_cr = min(coverage_ratios.values()), max(coverage_ratios.values())
        sort_key = (
            lambda c: coverage_ratios[c]
            + self._active_selection_randomness["sel_3"]
            * self.randomness
            * random.uniform(-1, 1)
            * (max_cr - min_cr)
            / 2
        )
        sorted_characteristics = sorted(self._instance.C, key=sort_key)
        for c in sorted_characteristics:
            if self._improve_characteristic(c, routes, route_durations, visited):
                return True
        return False

    def _define_solution_instance(self, routes) -> Solution:
        """
        Define a Solution instance from the routes.

        Args:
            routes: complete routes for each vehicle.

        Returns:
            Solution: Solution instance.
        """
        x = {
            (i, j, k): 0
            for i in self._instance.N_0
            for j in self._instance.N_0
            for k in self._instance.K
        }
        for k, route in routes.items():
            for i, j in zip(route[:-1], route[1:]):
                x[i, j, k] = 1

        objective = min(
            self._coverage_ratios(
                {i for route in routes.values() for i in route}
            ).values()
        )
        return Solution(self._instance, x, objective)

    def _improve_characteristic(self, c, routes, route_durations, visited) -> bool:
        """
        Improve the solution by inserting the closest node that carries characteristic c into the route where it would
        be the cheapest to insert it.

        Args:
            c: characteristic to improve.
            routes: partial routes for each vehicle.
            route_durations: current duration of each partial route.
            visited: set of visited nodes.

        Returns:
            bool: True if an improvement was done, False otherwise.
        """
        # Find the set of nodes that carry characteristic c and have not been visited yet.
        candidates = {
            i
            for i in self._instance.N
            if self._instance.alpha[i, c] and i not in visited
        }
        if not candidates:
            return False

        # Find the insertion cost: distance to the closest node in some partial route 'end'.
        # Additionally, save the route (k) where the insertion would be done.
        insertion_costs = {
            i: (
                min(self._instance.t[i, routes[k][-1]] for k in self._instance.K),
                min(self._instance.K, key=lambda k: self._instance.t[i, routes[k][-1]]),
            )
            for i in candidates
        }
        # Filter out nodes that cannot be inserted because current route duration + insertion cost + distance to
        # depot is greater than T_max.
        candidates = {
            i
            for i in candidates
            if route_durations[insertion_costs[i][1]]
            + insertion_costs[i][0]
            + self._instance.t[i, 0]
            <= self._instance.T_max
        }
        if not candidates:
            return False

        # SELECTION 4
        # Choose the node with the lowest insertion cost and insert it into the route
        min_ic, max_ic = (
            min(insertion_costs.values(), key=lambda x: x[0])[0],
            max(insertion_costs.values(), key=lambda x: x[0])[0],
        )
        sort_key = lambda i: insertion_costs[i][0] + self._active_selection_randomness[
            "sel_4"
        ] * self.randomness * random.uniform(-1, 1) * 0.5 * (max_ic - min_ic)
        i = min(candidates, key=sort_key)
        self.add_i_to_route_of_k(
            i, insertion_costs[i][1], route_durations, routes, visited
        )

        return True

    def add_i_to_route_of_k(self, i, k, route_durations, routes, visited) -> None:
        """
        Add node i to the partial route of vehicle k.

        Args:
            i: node to add.
            k: vehicle to add node i to.
            route_durations: current duration of each partial route.
            routes: partial routes for each vehicle.
            visited: set of visited nodes.
        """
        route_durations[k] += self._instance.t[routes[k][-1], i]
        routes[k].append(i)
        visited.add(i)

    def run_heuristics(self, time_limit: int, n_solutions: int = 1) -> List[Solution]:
        """
        Generate feasible solutions using the greedy heuristic for a given time limit. The solutions are generated
        using different levels of randomness, and the best n_solutions are returned. Some different levels of randomness
        and randomness activations are tried, and the choice for them is based on the best results.

        Args:
            time_limit: time limit in seconds.
            n_solutions: number of solutions to return.

        Returns:
            List[Solution]: List of feasible solutions.
        """
        start = time.time()
        print("\nRunning heuristics to find feasible solutions.")
        spaces = [
            (np.linspace(0, 2, 75), {"sel_1": 0, "sel_2": 0, "sel_3": 1, "sel_4": 0}),
            (
                np.linspace(0, 2, 20)[1:],
                {"sel_1": 1, "sel_2": 0, "sel_3": 1, "sel_4": 0},
            ),
            (
                np.linspace(0, 2, 20)[1:],
                {"sel_1": 1, "sel_2": 1, "sel_3": 1, "sel_4": 1},
            ),
            (
                np.linspace(0, 2, 20)[1:],
                {"sel_1": 1, "sel_2": 0, "sel_3": 1, "sel_4": 1},
            ),
            (
                np.linspace(0, 4, 1000)[1:],
                {"sel_1": 1, "sel_2": 1, "sel_3": 1, "sel_4": 1},
            ),
        ]
        solutions = []
        for i, (randomness, active_selection_randomness) in enumerate(spaces):
            print(f" -- Running batch {i + 1}")
            self._active_selection_randomness = active_selection_randomness
            for r in randomness:
                if time.time() - start > time_limit:
                    break
                sol = self.build_solution(r, seed=int(r * 1000))
                solutions.append((sol, sol.obj))

        print(f"{len(solutions)} randomized GH solutions were explored.")
        solutions = sorted(solutions, key=lambda x: x[1], reverse=True)
        solutions = list(dict.fromkeys(solutions))[0:n_solutions]
        solutions = [sol[0] for sol in solutions]
        print(f"Best heuristic solution: {round(solutions[0].obj, 3)}\n")

        return solutions


def test_gh():
    """
    Test the greedy heuristic on the large instances. The heuristic is run 30 seconds for each instance, and the best
    solution is returned. The results are compared to the results of the TS algorithm and the GH reported in the paper.
    """
    global obj
    config = Config()
    assert config.instance_type == "large", "Check config settings. Need type = large"
    instance_loader = InstanceLoader(config)
    instances = instance_loader.load_instances(id_indices=False)
    results = {}
    for instance_name, instance in list(instances.items()):
        print(f"\nInstance: {instance_name}")

        Z_TS = instance.instance_results["Z_TS"]
        Z_GH = Z_TS / (1 + instance.instance_results["ts_vs_gh_gap"] / 100)
        greedy_heuristic = GreedyHeuristic(instance)
        det_sol = greedy_heuristic.build_solution(0)
        solutions = greedy_heuristic.run_heuristics(30, 1)
        solution = solutions[0]

        results[instance_name] = (Z_TS, Z_GH, solution.obj, det_sol.obj)
        print(
            f"Z_TS = {Z_TS}, Z_GH = {round(Z_GH, 3)}, "
            f"Z_x = {round(solution.obj, 3)}, Z_det = {round(det_sol.obj, 3)}"
        )
    n_better, n_worse, beat_TS, random_wins_det, gap, gap_ts = 0, 0, 0, 0, 0, 0
    for instance_name, (Z_TS, Z_GH, obj, det_obj) in results.items():
        gap += (obj - Z_GH) / Z_GH if Z_GH > 0 else 0
        gap_ts += (obj - Z_TS) / Z_TS if Z_TS > 0 else 0
        if Z_GH > obj:
            n_worse += 1
        else:
            n_better += 1
        if Z_TS <= obj:
            beat_TS += 1
        if obj > det_obj:
            random_wins_det += 1
    print("----------- Results --------------")
    print(f"Z_GH > Z_x: {n_worse} / {len(results)} times")
    print(f"Z_TS <= Z_x: {beat_TS} / {len(results)} times")
    print(f"Z_x > Z_det: {random_wins_det} / {len(results)} times")
    print(f"AVG gap (Z_x - Z_GH) / Z_GH %: {round(100 * gap / len(results), 3)}")
    print(f"AVG gap (Z_x - Z_TS) / Z_TS %: {round(100 * gap_ts / len(results), 3)}")


if __name__ == "__main__":
    test_gh()
