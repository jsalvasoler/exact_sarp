from optimizer import Optimizer
from src.config import Config
from src.formulations.cutset_formulation import CutSetFormulation
from src.formulations.mtz_formulation import MTZFormulation
from src.formulations.mtz_opt_formulation import MTZOptFormulation
from src.formulations.scf_formulation import SCFFormulation
from src.instance_loader import InstanceLoader

import pandas as pd
from pyinstrument import Profiler

formulations = {
    'mtz': MTZFormulation,
    # 'cutset': CutSetFormulation,
    'scf': SCFFormulation,
    'mtz_opt': MTZOptFormulation,
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


def big_experiment():
    config = Config()
    assert config.n_instances_main is None and config.n_instances_big, \
        'In big experiment, instance number is provided by n_instances_big'
    assert config.instance_type == 'large', 'Big experiment is for large instances only'
    assert config.time_limit == 60, 'Big experiment needs time_limit to be 60'

    results = pd.read_csv(config.results_file, sep=';', decimal=',')

    # Find instances + formulations that have already been solved
    # That means either the solve_time is greater than the time_limit or the mip_gap is zero + tolerance
    solved = results.loc[
        (results['id'] < 49) & ((results['solve_time'] >= config.time_limit * 60) | (results['mip_gap'].abs() < 1e-6))
        , ['id', 'formulation']].values

    all_executions = {(instance_id, formulation_name) for instance_id in range(1, 49) for formulation_name in
                      formulations.keys()}
    to_execute = sorted(list(all_executions - set(map(tuple, solved))))
    print(f'All executions: {sorted(all_executions)}')
    print(f'To execute: {to_execute}')

    print(f'Already solved {len(solved)} / {4 * 48} problems\n')
    instance_loader = InstanceLoader(config)
    instances = instance_loader.load_instances()

    for i, (instance_id, formulation_name) in enumerate(to_execute[:config.n_instances_big]):
        config.set_formulation(formulation_name)
        instance = instances[instance_id]
        print(f'\nInstance {i + 1}/{config.n_instances_big}: \n  '
              f'id = {instance_id}, name = {instance.name}, formulation = {formulation_name}\n'
              f'instance information: {instance.instance_results}\n\n')
        formulation = formulations.get(config.formulation)(instance, config.activations.get(config.formulation, {}))
        optimizer = Optimizer(formulation, config)
        optimizer.run()


if __name__ == '__main__':
    profiler = Profiler()
    profiler.start()
    # main()
    big_experiment()
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
