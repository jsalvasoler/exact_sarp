# Exact methods for the SARP problem: Selective Assessment Routing Problem

This repository contains the implementation of the formulations and algorithms proposed in the research project **Exact methods for the SARP problem**. 

This repository is accompanied by the theoretical work available on the link below (preliminary version, expected to be finished by December 2023). Our main finding is that the proposed Single Commodity Flow formulation is theoretically stronger than the current best MTZ-3 formulation by B. Balcik.
- [Exact Methods for SARP.pdf](https://github.com/jsalva9/exact_sarp/blob/master/Theoretical%20Work%20and%20Results/Exact%20Methods%20for%20SARP.pdf)

The results on the literature benchmark instances obtained so far outperform in many cases the specialized state-of-the-art algorithms for SARP. All the results are available here:
- [Results on 112 benchmark instances](https://github.com/jsalva9/exact_sarp/blob/master/Theoretical%20Work%20and%20Results/all_results.csv)

The following is the abstract of the project.

### Abstract
In the immediate aftermath of a disaster, relief agencies perform rapid needs assessments to investigate the effects of the disaster on the affected 
communities. Since assessments must be performed quickly, visiting all of the sites in the affected region may not be possible. 
The Selective Assessment Routing Problem (SARP) was first defined by [Balcik, B.](https://www.sciencedirect.com/science/article/pii/S1366554516303106) and addresses the site selection and routing decisions 
of the rapid needs assessment teams which aim to evaluate the post-disaster conditions of different community groups, each carrying a distinct 
characteristic. SARP constructs an assessment plan that maximizes the covering of different characteristics in a balanced way. 
While some approximate solutions have been already proposed, this project explores exact approaches that 
maximally exploit or extend the capabilities of numerical solvers. Different mathematical formulations will be discussed, and different exact 
optimization techniques will be considered to improve the performance of numerical solvers.
