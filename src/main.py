from src.formulations.mtz_formulation import MTZFormulation
from optimizer import Optimizer
from src.config import Config
from utils import Instance
from src.instance_loader import InstanceLoader
from pyinstrument import Profiler


def test_single_instance():
    instance = Instance(75, 15, 50, 150, seed=0)
    config = Config()
    formulations = {
        'mtz': MTZFormulation(instance, config.activations.get('mtz', {})),
    }
    optimizer = Optimizer(formulations['mtz'], config)
    optimizer.run()


def main():
    config = Config()
    instance_loader = InstanceLoader(config)
    instances = instance_loader.load_instances()
    formulations = {
        'mtz': MTZFormulation,
    }
    for i, (name, instance) in enumerate(instances.items()):
        print(f'\nInstance {i + 1}/{len(instances)}: \n  id = {name[:2]}, name = {name[3:-4]}\n\n')
        instance.print()

        formulation = formulations.get(config.formulation)(instance, config.activations.get(config.formulation, {}))
        optimizer = Optimizer(formulation, config)
        optimizer.run()


if __name__ == '__main__':

    profiler = Profiler()
    profiler.start()
    # test_single_instance()
    main()
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
