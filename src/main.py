from src.formulations.mtz_formulation import MTZFormulation
from optimizer import Optimizer
from src.config import Config
from utils import Instance

if __name__ == '__main__':
    instance = Instance(75, 15, 50, 150, seed=0)
    config = Config()

    formulations = {
        'mtz': MTZFormulation(instance, config.activations.get('mtz', {})),
    }

    optimizer = Optimizer(formulations['mtz'], config)
    optimizer.run()
