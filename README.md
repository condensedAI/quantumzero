# Overview

Quantumzero aims to automate the design of annealing schedules in a hybrid quantum-classical framework. We study both MCTS and QZero algorithms's performance in discovering effective annealing schedules even when the annealing time is short for the 3-SAT examples.

# Software requirements

## OS Requirements
This package is supported for macOS and Windows. The package has been tested on the following systems:

macOS: Mojave (10.14.1)

windows 10

## Python Dependencies
quantumzero mainly depends on the Python scientific stack.

numpy

math

scipy

matplotlib

qutip


## Files:

problem.py: contains the function to define the Hamiltonian for the 3-SAT and computes initial and target states

annealer.py: contains the classes that implement the continuous and digitized annealing evolution. They have a general structure, so should work for any spin Hamiltonian where the target is diagonal in the z-basis

methods.py: contains the old stochastic gradient descent (deprecated), the monte carlo tree search wrapper for the specifc problem and the quantum approximate optimization algorithm

mcts.py: contains the class that implement the mcts algorithm with a general reward function that is passed in the method.py file

run-simulation.py: main program, runs all the optimization algorithm implemented so far. type python run-simulation.py --help for information on the input paramters

dataset/ : folder with the files used to create the 3-SAT hamiltonian system sizes from N=7 to N=15. If moved to a different location, it is necesssary to modify the path in run-simulation.py accordingly.
