# An ANN-surrogate based optimization framework
Instead of using adjoints from a numerical solver, we use the adjoints of a neural network to find optimal solutions to a cost function. Adjoints of the numerical solver may also be used to improve the accuracy of the NN. 

In addition, we may also train an RL agent to explore the search space of solutions. 

# Sample results

Optimization using adjoint of surrogate DNN

![DNN](https://github.com/Romit-Maulik/Practice/blob/master/projection_optimize/Shapes/Shape_Comparison_DNN.jpeg)


Optimization using RL + surrogate DNN

![RL](https://github.com/Romit-Maulik/Practice/blob/master/projection_optimize/Shapes/Shape_Comparison_RL.jpeg)
