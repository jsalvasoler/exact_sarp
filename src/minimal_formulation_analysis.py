from functools import cache
from pyinstrument import Profiler

from src.instance_loader import InstanceLoader
from src.config import Config
from src.optimizer import Optimizer
from src.main import formulations
from typing import Dict, Set, List


def minimal_formulation_analysis(formulation_name: str) -> None:
    """
    Find all minimal sets of constraints for a given formulation.
    Minimal in the sense that removing any constraint from the set makes the formulation invalid
    (optimal value is better than the real one).

    Args:
        formulation_name: name of the formulation to analyze.
    """

    assert not config.n_instances, f'Cannot run minimal_formulation_analysis with n_instances != None'
    instance_loader = InstanceLoader(config)
    all_instances = instance_loader.load_instances()
    instance_names = ['small_R12hom_imp3_K3T2', 'small_R12hom_alt2_K2T2', 'small_RC12het_altimpinc3_K3T4',
                      'small_R12het_altimpinc3_K2T3']
    instances = [v for k, v in all_instances.items() if any(name in k for name in instance_names)]
    assert len(instances) == len(instance_names), f'Something went wrong when filtering instances.'

    # Get set of all constraints
    temp_formulation = formulations.get(formulation_name)(instances[0])
    temp_formulation.formulate()
    all_constraints = set(temp_formulation.constraints.keys())

    @cache
    def minimal_set(current_constraints: str) -> List[str]:
        """
        Find all minimal sets of constraints given a set of current constraints.
        Algorithm: for each constraint, test if the formulation is valid without it. If it is, call minimal_set on the
        remaining constraints. If no constraint can be removed, return current_constraints is a minimal set.

        Args:
            current_constraints: set of current constraints (string representation). It is assumed to be valid.

        Returns:
            List of all minimal sets of constraints.
        """
        # Retrieve current constraints
        current_constraints = eval(current_constraints)

        # Build the activations dictionary
        activations = {c: True if c in current_constraints else False for c in all_constraints}

        ans = []
        for constraint in current_constraints:
            activations[constraint] = False
            if formulation_is_valid(str(activations)):
                ans += minimal_set(str(current_constraints - {constraint}))
            activations[constraint] = True

        if not ans:
            return [str(current_constraints)]
        return ans

    @cache
    def formulation_is_valid(activations: str) -> bool:
        """
        Checks if the formulation is valid for the given instances and activations.
        Valid means that the optimal value of the problem corresponds to the known one.

        Args:
            activations: dictionary of constraints to activate or not (string representation).

        Returns:
            True if the formulation is valid for the given instances and activations, False otherwise.
        """
        # Retrieve activations
        activations = eval(activations)

        assert config.exception_when_non_optimal, f'Cannot validate formulation without exception_when_non_optimal.'
        for instance in instances:
            formulation = formulations.get(formulation_name)(instance, activations)
            optimizer = Optimizer(formulation, config)
            try:
                optimizer.run()
            except ValueError as e:
                return False

        return True

    ans = set(minimal_set(str(all_constraints)))
    final_list = []
    for a in ans:
        final_list.append((len(eval(a)), eval(a)))
    print(f'All constraints: \n  {len(all_constraints)}: {str(all_constraints)}')
    print(f'Minimal sets:')
    for size, s in sorted(final_list):
        print(f'  {size}: {str(s)}')

    # Print the corresponding activations for the yaml file
    for i, a in enumerate(ans):
        print(f'Minimal set {i}\n{formulation_name}:')
        for c in all_constraints:
            print(f'  {c}: {"true" if c in eval(a) else "false"}')


if __name__ == '__main__':
    prof = Profiler()
    prof.start()

    config = Config()
    minimal_formulation_analysis(config.formulation)

    prof.stop()
    print(prof.output_text(unicode=True, color=True))
