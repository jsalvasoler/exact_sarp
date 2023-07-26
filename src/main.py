from src.formulations.mtz_formulation import MTZFormulation
from src.formulations.cutset_formulation import CutSetFormulation
from src.formulations.scf_formulation import SCFFormulation
from optimizer import Optimizer
from src.config import Config
from src.instance_loader import InstanceLoader
from pyinstrument import Profiler

formulations = {
    'mtz': MTZFormulation,
    'cutset': CutSetFormulation,
    'scf': SCFFormulation,
}


def main():
    config = Config()
    instance_loader = InstanceLoader(config)
    instances = instance_loader.load_instances()

    for i, (name, instance) in enumerate(instances.items()):
        print(f'\nInstance {i + 1}/{len(instances)}: \n  id = {name[:2]}, name = {name[3:-4]}\n\n')
        instance.print()

        formulation = formulations.get(config.formulation)(instance, config.activations.get(config.formulation, {}))
        optimizer = Optimizer(formulation, config)
        optimizer.run()


if __name__ == '__main__':

    profiler = Profiler()
    profiler.start()
    main()
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
