import numpy as np
np.random.seed(10)
from problem import shallow_water

if __name__ == '__main__':
    print('Running non-linear SWE')
    new_run = shallow_water()
    new_run.solve()