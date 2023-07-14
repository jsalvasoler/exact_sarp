from mtz_formulation import MTZFormulation
from optimizer import Optimizer
from utils import Instance

if __name__ == '__main__':
    instance = Instance(10, 2, 10, 10)

    activations = {
        'not_stay': True,
    }

    formulations = {
        'mtz': MTZFormulation(instance, activations),
    }

    optimizer = Optimizer(formulations['mtz'])
    optimizer.run()
