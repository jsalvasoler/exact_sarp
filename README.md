# Exact methods for the SARP problem: Selective Assessment Routing Problem

This repository contains the implementation of the formulations and algorithms proposed in the research project **Exact methods for the SARP problem**. 
The author of the code is Joan Salvà Soler, and Vera Hemmelmayr and Günther Raidl supervised the project. 
Find in [Exact Methods for SARP](https://link.springer.com/article/10.1007/s10100-024-00943-y?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=oa_20241015&utm_content=10.1007%2Fs10100-024-00943-y) the published version of the paper.

Our main finding is that the proposed Single Commodity Flow formulation is theoretically stronger than the current-best MTZ-3 formulation by B. Balcik. The results on the literature benchmark instances obtained so far outperform in many cases the specialized state-of-the-art algorithms for SARP. All the results are available here:
- [Benchmark results](https://github.com/jsalva9/exact_sarp/blob/master/results/all_results.csv)


### Install
Create a virtual environment with Python 3.10 and install the requirements.
```bash
python -m venv exact_sarp_env
source exact_sarp_env/bin/activate
pip install -r requirements.txt
```
The project uses the MIP solver Gurobi. You can get a free academic license [here](https://www.gurobi.com/academia/academic-program-and-licenses/).


### Run
To run the project, run `python src/main.py` with a suitable `config.yaml` file placed in the project root.

The following is an example configuration file:

```yaml
execution:
  instance_type: small  # small, large, case
  n_instances_main:     # number of instances to solve for the standard execution
  n_instances_big: 1        # number of instances to solve for the big execution
  instance_name:   # name of the instance to solve. Only if n_instances_main = 1
  seed: 2               # random seed used in the instance generator and selector
  exception_when_non_optimal: true # if true, the program will stop when a non-optimal solution is found

solver:
  time_limit: 60 # in minutes
  print_solution: 1 # 0, 1, or 2
  draw_solution: false

formulation: scf_sep_cuts # mtz, cutset, scf, mtz_opt. Additionally, scf_cuts_2, scf_cuts_3, scf_sep_cuts, scf_start

activations:  # whether a particular constraint is added to the formulation. If key is missing, assume constraint is added
  mtz:
    not_stay: true
    mtz: true
    number_of_vehicles_hard: true
    visit: true
    enter_depot: true
    define_obj: true
    max_time: true
    enter: true
    leave: true

  mtz_opt:
    arrival_time: true

  cutset:
    cutset_integer: true
    cutset_relaxation: false

```

### Instances
The project used the instances gathered by Balcik, B. They are available for public use in the `data` directory. This data set was developed as part of the following funded project.

Balcik, B. (Principal Investigator). (2014). Selective Routing Problems for Post-Disaster Needs Assessment: Models and Solution Methods. The Scientific and Technological Research Council of Türkiye (TÜBİTAK). Grant No. 213M414.


### Abstract
The Selective Assessment Routing Problem (SARP) addresses the site selection
and routing decisions of rapid needs assessment teams which aim to evaluate the
post-disaster conditions of different community groups, each carrying a distinct
characteristic. SARP constructs an assessment plan that maximizes the covering
of different characteristics in a balanced way. We explore exact approaches based
on mixed integer linear programming: different mathematical formulations are
presented, and theoretical results regarding their strength are derived. The models are experimentally evaluated on a set of test instances, and the best model is
applied to a real-world scenario.


