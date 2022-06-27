# Overview

The repository contains code for optimizing annealing schedules in a hybrid quantum-classical framework. It is based off of https://github.com/yutuer21/quantumzero, and accompanies a reusability report for Nature Machine Intelligence. We extend the results to include a BFGS gradient method comparison, and to include a MaxCut problem set.

## Description:

The best starting point for this codebase is to look at the `Demo.ipynb` jupyter notebook. It contains demonstrations of how the different components of the code work. The `run-simulation.py` file is then used as a standalone to run and produce data for the plots in the accompanying Nature Machine Intelligence Reusability Report. 

This repository also includes a second version of the MCTS code (indicated by `_v2` in method names), though the results for the report were all obtained using the original (for reusability reasons).
