# An ANN-surrogate based optimization framework for the RAE2822 transonic airfoil.
Instead of using adjoints from a numerical solver, we use the adjoints of a neural network to find optimal solutions to a cost function.

In addition, we may also train an RL agent to explore the search space of solutions. 

# To train a surrogate DNN model
```
python nn_opt.py train shape
```
# To use trained model for shape optimization
```
python nn_opt.py optimize shape
```
# To use trained model for lift constrained shape optimization
```
python nn_opt.py optimize shape_lift
```

# Sample results

In `Shapes/` folder