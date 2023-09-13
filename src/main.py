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
    'cutset': CutSetFormulation,
    'scf': SCFFormulation,
    'mtz_opt': MTZOptFormulation,
}


def main():
    config = Config()
    instance_loader = InstanceLoader(config)
    instances = instance_loader.load_instances(id_indices=False)

    for i, (name, instance) in enumerate(instances.items()):
        print(f'\nInstance {i + 1}/{len(instances)}: \n  id = {name[:2]}, name = {name[3:-4]}\n\n')
        instance.print()

        formulation = define_formulation(config, instance)
        optimizer = Optimizer(formulation, config)
        optimizer.run()


def define_formulation(config, instance):
    if 'scf' in config.formulation:
        formulation = formulations.get('scf')(
            instance, config.activations.get(config.formulation, {}), variant=config.formulation)
    else:
        formulation = formulations.get(config.formulation)(instance, config.activations.get(config.formulation, {}))
    return formulation


def big_experiment():
    config = Config()
    config.results_file = 'big_results.csv'

    assert config.n_instances_main is None and config.n_instances_big, \
        'In big experiment, instance number is provided by n_instances_big'
    assert config.time_limit == 60, 'Big experiment needs time_limit to be 60'

    results = pd.read_csv(config.results_filepath, sep=';', decimal=',')

    # Find instances + formulations that have already been solved
    # That means either the solve_time is greater than the time_limit or the mip_gap is zero + tolerance
    solved = results.loc[
        (results['type'] == 'large') &
        ((results['solve_time'] >= config.time_limit * 60) | (results['mip_gap'].abs() < 1e-6))
        , ['id', 'formulation']].values

    # ids = [1, 27, 11, 7, 14, 15, 23, 24] + [49, 58, 63, 69]
    ids = list(range(1, 97))
    form_names = ['scf_cuts_2_start']

    all_executions = {(instance_id, form_name) for instance_id in ids for form_name in form_names}
    to_execute = sorted(list(all_executions - set(map(tuple, solved))), key=lambda x: x[0])
    print(f'All executions: {sorted(all_executions)}')
    print(f'To execute: {to_execute}')

    instance_loader = InstanceLoader(config)
    instances = instance_loader.load_instances()

    for i, (instance_id, formulation_name) in enumerate(to_execute[:config.n_instances_big]):
        config.set_formulation(formulation_name)
        instance = instances[instance_id]
        print(f'\nInstance {i + 1}/{config.n_instances_big}: \n  '
              f'id = {instance_id}, name = {instance.name}, formulation = {formulation_name}\n'
              f'instance information: {instance.instance_results}\n\n')

        formulation = define_formulation(config, instance)
        optimizer = Optimizer(formulation, config)
        optimizer.run()


def instance_difficulty_experiment():
    instance_name, instance_id = 'large_RC50_K4T5', 42
    # instance_name, instance_id = 'large_RC25_K2T3', 8
    config = Config()
    assert config.instance_type == 'large' and config.formulation == 'scf', 'Check config settings. Need large + scf'
    config.results_large = 'results_instance_diff.csv'

    instance_loader = InstanceLoader(config)
    instances = instance_loader.load_instances()

    instance = instances[instance_id]

    T_max_set = [1, 2, 3, 4, 5, 6, 7]
    K_size_set = [6]
    to_execute = [(T_max, K_size) for T_max in T_max_set for K_size in K_size_set]
    for i, (T_max, K_size) in enumerate(to_execute):
        instance.update_T_max(T_max)
        instance.update_K_size(K_size)

        print(f'\nInstance {i + 1}/{len(to_execute)}: \n  T_max = {T_max}, K_size = {K_size}\n\n')
        instance.print()

        formulation = formulations.get(config.formulation)(instance, config.activations.get(config.formulation, {}))
        optimizer = Optimizer(formulation, config)
        optimizer.run()

    return None


if __name__ == '__main__':
    profiler = Profiler()
    profiler.start()
    # main()
    big_experiment()
    # instance_difficulty_experiment()
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
