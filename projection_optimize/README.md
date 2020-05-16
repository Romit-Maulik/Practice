# An ANN-surrogate based optimization framework
Instead of using adjoints from a numerical solver, we use the adjoints of a neural network to find optimal solutions to a cost function. Adjoints of the numerical solver may also be used to improve the accuracy of the NN. 

In addition, we may also train an RL agent to explore the search space of solutions. 

## To run
1. Training a regular surrogate model
`python nn_opt.py train regular` 
2. Training a surrogate model "enhanced" by adjoint information
`python nn_opt.py train augmented` 
3. Optimizing for minimum objective function (for e.g. lift to drag ratio) by using the regular surrogate model
`python nn_opt.py optimize regular`
4.  Optimizing for minimum objective function (for e.g. lift to drag ratio) by using the enhanced surrogate model
`python nn_opt.py optimize augmented`
5. Training a reinforcement learning agent using the regular surrogate model
`python nn_opt.py rl_train regular`
6. Training a reinforcement learning agent using the augmented surrogate model
`python nn_opt.py rl_train augmented`
7. Using reinforcement learning agent to find optimal shape using the regular model for episode evaluation
`python nn_opt.py rl_optimize regular`
8. Using reinforcement learning agent to find optimal shape using the augmented model for episode evaluation
`python nn_opt.py rl_optimize augmented`

## Work in progress
1. Adding constrained optimizers
2. Improving Bernstein polynomial representation (potentially bug in shape plotting)