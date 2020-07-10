# An ANN-surrogate based optimization framework
Instead of using adjoints from a numerical solver, we use the adjoints of a neural network to find optimal solutions to a cost function. Adjoints of the numerical solver may also be used to improve the accuracy of the NN. 

In addition, we may also train an RL agent to explore the search space of solutions. 