import random
import time
from typing import Tuple, List
from tqdm import tqdm
from pyinstrument import Profiler
import matplotlib.pyplot as plt

import numpy as np

from instance_loader import InstanceLoader
from utils import Instance, Solution
from config import Config


class RLHeuristic:
    """
    TODO: Add description
    """

    def __init__(
        self, instance: Instance, n_episodes: int, epsilon: int = 0.2, seed: int = 42
    ):
        random.seed(seed)
        self._instance = instance

        self._Q = {
            (i, j): 0 for i in self._instance.N_0 for j in self._instance.N_0 if i != j
        }  # Q matrix

        self._n_episodes = n_episodes
        self._epsilon = epsilon
        self._step_size = 0.1

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

    def run(self) -> Solution:
        ep_results = []
        best_sol_obj = -1
        best_sol_routes = None

        for _ in tqdm(range(self._n_episodes)):
            routes, cov_ratio = self._run_episode(epsilon=self._epsilon)
            ep_results.append(cov_ratio)

            if cov_ratio > best_sol_obj:
                best_sol_routes = routes
                best_sol_obj = cov_ratio

        # Plot cov_ratio over episodes and add 250-MA
        plt.plot(ep_results)
        ma = np.convolve(ep_results, np.ones(250) / 250, mode="valid")
        plt.plot(ma, label="250-MA")
        plt.legend()
        plt.title("Coverage ratio over episodes - instance " + self._instance.full_name)
        plt.xlabel("Episode")
        plt.ylabel("Objective")
        plt.show()

        return self._define_solution_instance(best_sol_routes)

    def _run_episode(self, epsilon):
        visited = set()
        routes = {k: [0] for k in self._instance.K}
        route_durations = {k: 0 for k in self._instance.K}

        current_node = 0
        cov_ratios = {c: 0 for c in self._instance.C}
        for k in self._instance.K:
            while True:
                candidates = {
                    i
                    for i in self._instance.N
                    if i not in visited
                    and route_durations[k]
                    + self._instance.t[current_node, i]
                    + self._instance.t[i, 0]
                    <= self._instance.T_max
                }
                if not candidates:
                    self.add_i_to_route_of_k(0, k, route_durations, routes, visited)
                    current_node = 0
                    break
                i = self._select_next_node(current_node, candidates, epsilon)
                self.add_i_to_route_of_k(i, k, route_durations, routes, visited)
                self.update_Q_and_cov_ratio(cov_ratios, visited, current_node, i)
                current_node = i
        return routes, min(cov_ratios.values())

    def update_Q_and_cov_ratio(self, cov_ratios, visited, current_node, i):
        old_cov_ratio = min(cov_ratios.values())
        for c in self._instance.C:
            if self._instance.alpha[i, c]:
                cov_ratios[c] += 1 / self._instance.tau[c]

        new_cov_ratio = min(cov_ratios.values())
        if new_cov_ratio > old_cov_ratio:  # reward for improving the coverage ratio
            reward = new_cov_ratio - old_cov_ratio
        else:  # if not, reward low travel times
            travel_time = self._instance.t[current_node, i]
            reward = -1 * travel_time / self._instance.T_max
            
        self._Q[current_node, i] += self._step_size * (
            reward
            + max(self._Q[i, j] for j in self._instance.N_0 if i != j)
            - self._Q[current_node, i]
        )

    def _select_next_node(self, current_node, candidates, epsilon):
        """
        Select the next node to visit.

        Args:
            current_node: current node.
            candidates: set of candidate nodes.

        Returns:
            int: Next node to visit.
        """
        if random.random() < self._epsilon:
            return random.choice(list(candidates))
        return max(candidates, key=lambda i: self._Q[current_node, i])

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


def test_rl():
    """
    Test the RL heuristic on the large instances. The heuristic is run 30 seconds for each instance, and the best
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
        rl_heuristic = RLHeuristic(instance, n_episodes=10000)
        solution = rl_heuristic.run()

        results[instance_name] = (Z_TS, Z_GH, solution.obj)
        print(
            f"Z_TS = {Z_TS}, Z_GH = {round(Z_GH, 3)}, "
            f"Z_x = {round(solution.obj, 3)}"
        )
    n_better, n_worse, beat_TS, random_wins_det, gap, gap_ts = 0, 0, 0, 0, 0, 0
    for instance_name, (Z_TS, Z_GH, obj) in results.items():
        gap += (obj - Z_GH) / Z_GH if Z_GH > 0 else 0
        gap_ts += (obj - Z_TS) / Z_TS if Z_TS > 0 else 0
        if Z_GH > obj:
            n_worse += 1
        else:
            n_better += 1
        if Z_TS <= obj:
            beat_TS += 1
    print("----------- Results --------------")
    print(f"Z_GH > Z_x: {n_worse} / {len(results)} times")
    print(f"Z_TS <= Z_x: {beat_TS} / {len(results)} times")
    print(f"AVG gap (Z_x - Z_GH) / Z_GH %: {round(100 * gap / len(results), 3)}")
    print(f"AVG gap (Z_x - Z_TS) / Z_TS %: {round(100 * gap_ts / len(results), 3)}")


if __name__ == "__main__":
    profiler = Profiler()
    profiler.start()
    test_rl()
    profiler.stop()
    print(profiler.output_text(color=True, unicode=True))