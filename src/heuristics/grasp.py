import random
import time
from typing import Tuple, List
from tqdm import tqdm
from pyinstrument import Profiler
import matplotlib.pyplot as plt
from copy import deepcopy

import numpy as np

from instance_loader import InstanceLoader
from utils import Instance, Solution
from config import Config
from heuristics.greedy_heuristic import GreedyHeuristic


class GRASP:
    """
    Greedy Randomized Adaptive Search Procedure (GRASP) for the SARP.
    The heuristic is based on the Randomized Greedy Heuristic, with the addition of a local search step.
    """

    def __init__(
            self, instance: Instance, n_trials: int, seed: int = 42
    ):
        random.seed(seed)
        self._instance = instance
        self._greedy_heuristic = GreedyHeuristic(instance)

        self._best_solution = None
        self._best_obj = -float("inf")

        self._randomness = 0.5
        self._step_function = 'best_improvement'  # in ['best_improvement', 'first_improvement', 'random']
        self._max_neighbors_explored = 25
        self._n_local_search_iterations = 25
        self._n_trials = n_trials

    def run_grasp_after_sampling(self, time_limit: int, solution_pool: list = None) -> Solution:
        if solution_pool is None:
            greedy_solutions = self._greedy_heuristic.run_heuristics(time_limit, self._n_trials * 3)
        else:
            greedy_solutions = sorted(solution_pool, key=lambda x: x.obj, reverse=True)
        # Keep n_trials best solutions having all different costs
        solution_pool = [greedy_solutions[0]]
        current_cost = greedy_solutions[0].obj
        for solution in greedy_solutions[1:]:
            if solution.obj != current_cost:
                solution_pool.append(solution)
                current_cost = solution.obj
            if len(solution_pool) == self._n_trials:
                break
        self._best_solution = deepcopy(solution_pool[0])
        self._best_obj = self._best_solution.obj

        for solution in tqdm(solution_pool):
            assert self._best_solution.obj == self._best_obj
            # print(f'Iteration {solution_pool.index(solution) + 1} / {len(solution_pool)}')
            # print(f'Current solution cost: {self._best_solution.obj}')
            # print(f' --------------------')
            local_search_solution = self._local_search(solution)
            if local_search_solution.obj > self._best_obj:
                self._best_obj = local_search_solution.obj
                self._best_solution = local_search_solution


        print(f'GRASP Best solution cost: {round(self._best_solution.obj, 3)}')
        return self._best_solution

    def run_original_grasp(self) -> Solution:
        for i in range(self._n_trials):
            starting_solution = self._greedy_heuristic.build_solution(self._randomness)
            local_search_solution = self._local_search(starting_solution)

            if local_search_solution.obj > self._best_obj:
                self._best_obj = local_search_solution.obj
                self._best_solution = local_search_solution

        return self._best_solution

    def _local_search(self, solution: Solution) -> Solution:
        if solution.obj == 1.0:
            return solution
        incumbent = solution
        for _ in range(self._n_local_search_iterations):
            neighbor_type = random.choice(solution.neighbor_types)
            iteration = 0
            while iteration < self._max_neighbors_explored:
                iteration += 1
                neighbor = solution.generate_neighbor(type=neighbor_type)
                if neighbor.obj > incumbent.obj or random.random() < 0.2:
                    incumbent = neighbor
        return incumbent


if __name__ == "__main__":
    profiler = Profiler()
    profiler.start()

    instance_loader = InstanceLoader(Config())
    instances = instance_loader.load_instances(id_indices=False)
    instance = list(instances.values())[0]

    grasp = GRASP(instance, n_trials=20)
    # grasp.run_original_grasp()
    grasp.run_grasp_after_sampling(5)
    profiler.stop()
    print(profiler.output_text(color=True, unicode=True))
