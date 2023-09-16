import os
import random
import pandas as pd
from typing import List, Dict
from src.config import Config
from src.utils import Instance, FIELDS_INSTANCE_RESULTS


class InstanceLoader:
    """
    This class is responsible for loading the type of instances defined in the config file. If there are filters on
    the number of instances, it will also apply them.

    Attributes:
        __config: The config object
        __instance_dir: The directory where the instances are stored
        __optimal_solutions_small: The optimal solutions for the small instances
        __original_results_large: The original results for the large instances
        __original_results_case: The original results for the case instances
    """
    def __init__(self, config: Config):
        self.__config = config
        self.__instance_dir = os.path.join(self.__config.data_dir, self.__config.instance_type)

        self.__optimal_solutions_small = self.read_optimal_solutions_small() \
            if self.__config.instance_type == 'small' else None
        self.__original_results_large = self.read_original_results_large() \
            if self.__config.instance_type == 'large' else None
        self.__original_results_case = self.read_original_results_case() \
            if self.__config.instance_type == 'case' else None

    def read_optimal_solutions_small(self) -> Dict[str, float]:
        """
        Read the optimal solutions for the small instances from the file

        Returns:
            Dict[str, float]: The optimal solutions
        """
        with open(self.__config.optimal_solutions_file, 'r') as f:
            lines = f.readlines()
            optimal_solutions = {}
            for line in lines:
                name, inst_id, value = line.replace('\n', '').split(' ')
                value = float(value.replace(',', '.'))
                optimal_solutions[name] = value
                optimal_solutions[inst_id] = value
        return optimal_solutions

    def read_original_results_large(self) -> pd.DataFrame:
        """
        Read the original results for the large instances from the file

        Returns:
            pd.DataFrame: The original results
        """
        return pd.read_csv(self.__config.large_original_results_file, sep=';', decimal=',')

    def read_original_results_case(self) -> pd.DataFrame:
        """
        Read the original results for the case instances from the file

        Returns:
            pd.DataFrame: The original results
        """
        return pd.read_csv(self.__config.case_original_results_file, sep=';', decimal=',')

    def load_instances(self, id_indices: bool = True) -> Dict[str, Instance]:
        """
        Load the instances from the directory. If id_indices is True, the keys of the dictionary will also be ids.
        Otherwise, the keys will be the only instance names.

        Args:
            id_indices: Whether to use include ids as keys or not

        Returns:
            Dict[str, Instance]: The instances
        """
        # List all instances in the directory
        instance_names = os.listdir(self.__instance_dir)

        # Take just the first instance for the test
        instances = {}

        for i, instance_name in enumerate(instance_names):
            instance = self.__load_instance(instance_name)
            instances[str(instance_name)] = instance
            if id_indices:
                instances[instance.id] = instance

        if self.__config.n_instances_main is not None:

            if self.__config.n_instances_main == 1 and self.__config.instance_name:
                return {k: v for k, v in instances.items() if v.name == self.__config.instance_name}
            # Select randomly n_instances instances using random.sample
            random.seed(self.__config.seed)
            instances = {k: v for k, v in instances.items()}
            instances = dict(random.sample(instances.items(), self.__config.n_instances_main))

        return instances

    def __load_instance(self, instance_file: str) -> Instance:
        """
        Load a single instance from the file. It implements the different logic for the three instance types (small,
        large, case).

        Args:
            instance_file: The name of the instance file to load

        Returns:
            Instance: The instance object
        """
        instance_path = os.path.join(self.__instance_dir, instance_file)
        with open(instance_path, 'r') as f:
            lines = f.readlines()

            if self.__config.instance_type in ['small', 'large']:

                T_max = int(lines[0])
                factor = int(lines[1])
                K_size = int(lines[2])
                N_size = int(lines[3])
                C_size = int(lines[4])

                node_info = [tuple(map(int, line.split())) for line in lines[5:5 + N_size + 1]]
                coordinates = {i: (x[0], x[1]) for i, x in enumerate(node_info)}

                def euclidean_dist(i, j):
                    return ((coordinates[i][0] - coordinates[j][0]) ** 2 +
                            (coordinates[i][1] - coordinates[j][1]) ** 2) ** 0.5

                t = {(i, j): euclidean_dist(i, j) / factor for i in coordinates.keys() for j in coordinates.keys()}
                alpha = {(i + 1, c + 1): node_info[i + 1][2 + c] for c in range(C_size) for i in range(N_size)}

            else:
                T_max = int(lines[0])
                K_size = int(lines[1])
                N_size = int(lines[2])
                C_size = int(lines[3])

                node_info = [tuple(map(int, line.split())) for line in lines[4:4 + N_size + 1]]
                travel_times = [tuple(map(int, line.split())) for line in lines[4 + N_size + 1:4 + 2 * (N_size + 1)]]

                t = {(i, j): travel_times[i][j] for i in range(N_size + 1) for j in range(N_size + 1)}
                alpha = {(i + 1, c + 1): node_info[i + 1][c] for c in range(C_size) for i in range(N_size)}

        if self.__config.instance_type == 'small':
            instance_results = {
                'optimal_value': self.__optimal_solutions_small.get(
                    instance_file[3:-4], KeyError('Instance not found in optimal solutions'))
            }
        elif self.__config.instance_type == 'large':
            instance_results = self.get_instance_results_for_large(instance_file, N_size, K_size, T_max)
        else:       # case
            instance_results = self.get_instance_results_for_case(instance_file, C_size, K_size)

        return Instance(N_size, K_size, T_max, C_size, t, alpha, full_name=instance_file,
                        instance_results=instance_results)

    def get_instance_results_for_case(self, instance_file, C_size, K_size) -> Dict[str, float]:
        """
        Get the instance results for the particular instance from the case results dataframe

        Args:
            instance_file: name of the instance file
            C_size: number of characteristics
            K_size: number of vehicles

        Returns:
            Dict[str, float]: The instance results
        """
        T_max = int(instance_file.split('T')[1][:-4])
        df_row = self.__original_results_case.loc[
            (self.__original_results_case['C_size'] == C_size) &
            (self.__original_results_case['T_max'] == T_max) &
            (self.__original_results_case['K_size'] == K_size),
            FIELDS_INSTANCE_RESULTS]
        assert len(df_row) == 1, 'Error in extracting instance results'

        # Convert to dict and return
        return df_row.to_dict(orient='records')[0]

    def get_instance_results_for_large(self, instance_file, N_size, K_size, T_max) -> Dict[str, float]:
        """
        Get the instance results for the particular instance from the large results dataframe

        Args:
            instance_file: name of the instance file
            N_size: number of nodes
            K_size: number of vehicles
            T_max: maximum route duration

        Returns:
            Dict[str, float]: The instance results
        """
        network_type = 'RC' if 'RC' in instance_file else 'R'
        df_row = self.__original_results_large.loc[
            (self.__original_results_large['network_type'] == network_type) &
            (self.__original_results_large['nodes'] == N_size) &
            (self.__original_results_large['T_max'] == T_max) &
            (self.__original_results_large['K_size'] == K_size),
            FIELDS_INSTANCE_RESULTS]
        assert len(df_row) == 1, 'Error in extracting instance results'

        # Convert to dict and return
        return df_row.to_dict(orient='records')[0]
