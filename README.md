# Exact methods for the SARP problem: Selective Assessment Routing Problem

This repository contains the implementation of the formulations and algorithms proposed in the research project **Exact methods for the SARP problem**. 
The author of the code is Joan Salvà Soler, and Vera Hemmelmayr and Günther Raidl supervised the project. 

This repository is accompanied by a full report on the project results, which is available on the link below. Our main finding is that the proposed Single Commodity Flow formulation is theoretically stronger than the current best MTZ-3 formulation by B. Balcik.
- [Exact Methods for SARP.pdf](https://github.com/jsalvasoler/exact_sarp/blob/master/Theoretical%20Work%20and%20Results/Exact%20Methods%20for%20the%20Selective%20Assessment%20Routing%20Problem.pdf)

The results on the literature benchmark instances obtained so far outperform in many cases the specialized state-of-the-art algorithms for SARP. All the results are available here:
- [Benchmark results](https://github.com/jsalva9/exact_sarp/blob/master/Theoretical%20Work%20and%20Results/all_results.csv)

The following is the abstract of the project.

### Install
Create a virtual environment with Python 3.10 and install the requirements.
```bash
python -m venv exact_sarp_env
source exact_sarp_env/bin/activate
pip install -r requirements.txt
```
The project uses the MIP solver Gurobi. You can get a free academic license [here](https://www.gurobi.com/academia/academic-program-and-licenses/).


### Abstract
The Selective Assessment Routing Problem (SARP) addresses the site selection
and routing decisions of rapid needs assessment teams which aim to evaluate the
post-disaster conditions of different community groups, each carrying a distinct
characteristic. SARP constructs an assessment plan that maximizes the covering
of different characteristics in a balanced way. We explore exact approaches based
on mixed integer linear programming: different mathematical formulations are
presented, and theoretical results regarding their strength are derived. The models are experimentally evaluated on a set of test instances, and the best model is
applied to a real-world scenario.
