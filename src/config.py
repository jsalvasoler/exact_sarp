import yaml
import os


class Config:
    def __init__(self):
        # Get the root directory of the project. It is one upper level of the directory where this file is located.
        self.__root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.__config = self.__read_config()

        self._results_file = "results.csv"

    def __read_config(self):
        """
        Read .yaml config file and return a dictionary with the configuration.

        Returns:
            dict: Configuration dictionary.
        """
        with open(os.path.join(self.__root, "config.yaml"), "r") as f:
            config = yaml.safe_load(f)
        return config

    def set_formulation(self, formulation):
        self.__config["formulation"] = formulation

    @property
    def time_limit(self):
        return self.__config["solver"]["time_limit"]

    @property
    def activations(self):
        return self.__config["activations"]

    @property
    def print_solution(self):
        return self.__config["solver"]["print_solution"]

    @property
    def draw_solution(self):
        return self.__config["solver"]["draw_solution"]

    @property
    def instance_type(self):
        return self.__config["execution"]["instance_type"]

    @property
    def n_instances_main(self):
        return self.__config["execution"]["n_instances_main"]

    @property
    def n_instances_big(self):
        return self.__config["execution"]["n_instances_big"]

    @property
    def formulation(self):
        return self.__config["formulation"]

    @property
    def data_dir(self):
        return os.path.join(self.__root, "data")

    @property
    def results_filepath(self):
        return os.path.join(self.__root, "results", self._results_file)

    @property
    def optimal_solutions_file(self):
        return os.path.join(self.__root, "data", "optimal_solutions_small.txt")

    @property
    def large_original_results_file(self):
        return os.path.join(self.__root, "data", "large_instances_original_results.csv")

    @property
    def case_original_results_file(self):
        return os.path.join(self.__root, "data", "case_instances_original_results.csv")

    @property
    def seed(self):
        return self.__config["execution"]["seed"]

    @property
    def exception_when_non_optimal(self):
        return self.__config["execution"]["exception_when_non_optimal"]

    @property
    def instance_name(self):
        return self.__config["execution"]["instance_name"]

    @property
    def results_file(self):
        return self._results_file

    @results_file.setter
    def results_file(self, value):
        self._results_file = value
