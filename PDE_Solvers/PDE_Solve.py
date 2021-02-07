import numpy as np
np.random.seed(10)
from problems import shallow_water
from problems import burgers

if __name__ == '__main__':
    new_run = shallow_water()
    new_run.solve()