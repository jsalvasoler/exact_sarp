import yaml
import os


class Config:
    def __init__(self):
        # Get the root directory of the project. It is one upper level of the directory where this file is located.
        self.__root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.__config = self.__read_config()

    def __read_config(self):
        """
        Read .yaml config file and return a dictionary with the configuration.

        Returns:
            dict: Configuration dictionary.
        """
        with open(os.path.join(self.__root, 'config.yaml'), 'r') as f:
            config = yaml.safe_load(f)
        return config

    @property
    def time_limit(self):
        return self.__config['solver']['time_limit']

    @property
    def activations(self):
        return self.__config['activations']