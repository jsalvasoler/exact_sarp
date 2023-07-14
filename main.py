from mtz_formulation import MTZFormulation
from optimizer import Optimizer
from utils import Formulation, Instance

if __name__ == '__main__':
    instance = Instance(5, 2, 10, 3)

    activations = {
        'define': True, 'leave': True, 'enter': True, 'visit': True, 'leave_depot': True, 'max_time': True
    }

    formulations = {
        'mtz': MTZFormulation(instance, activations),
    }

    optimizer = Optimizer(formulations['mtz'])
    optimizer.run()
