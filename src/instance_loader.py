import os
import random
from typing import List, Dict
from src.config import Config
from src.utils import Instance


class InstanceLoader:
    def __init__(self, config: Config):
        self.__config = config
        self.__instance_dir = os.path.join(self.__config.data_dir, self.__config.instance_type)

    def load_instances(self) -> Dict[str, Instance]:
        # List all instances in the directory
        instance_names = os.listdir(self.__instance_dir)

        # Take just the first instance for the test
        instances = {}

        for i, instance_name in enumerate(instance_names):
            instance = self.__load_instance(instance_name)
            instances[str(instance_name)] = instance

        if self.__config.n_instances is not None:
            # Select randomly n_instances instances using random.sample
            instances = dict(random.sample(instances.items(), self.__config.n_instances))

        return instances

    def __load_instance(self, instance_file):
        instance_path = os.path.join(self.__instance_dir, instance_file)
        with open(instance_path, 'r') as f:
            lines = f.readlines()

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

            t = {(i, j): euclidean_dist(i, j) / factor
                 for i in coordinates.keys()
                 for j in coordinates.keys()
                 }

            alpha = {
                (i + 1, c + 1): node_info[i + 1][2 + c]
                for c in range(C_size)
                for i in range(N_size)
            }

            return Instance(N_size, K_size, T_max, C_size, t, alpha, full_name=instance_file)
