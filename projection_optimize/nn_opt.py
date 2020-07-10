import argparse
import tensorflow as tf
tf.random.set_seed(10)

import numpy as np
np.random.seed(15)
import matplotlib.pyplot as plt

from surrogate_models import coefficient_model, coefficient_model_adjoint
from optimizers import surrogate_optimizer
from utils import shape_return

if __name__ == '__main__':

    '''
    Usage: python nn_opt.py [train/optimize/rl_train/rl_optimize] [regular/augmented]
    '''

    parser = argparse.ArgumentParser(description='Adjoint augmented surrogate model based optimization')
    parser.add_argument('mode', metavar='mode', type=str, help='[train/optimize/rl_optimize]') #Train a network or use a trained network for minimize/rl_minimize
    parser.add_argument('constraint_options',metavar='constraint_options', type=str, help='[shape/shape_lift]')
    parser.add_argument('adjoint_mode', metavar='adjoint_mode', type=str, help='[regular/augmented]')
    args = parser.parse_args()

    # Load dataset
    input_data = np.load('DOE_2000.npy').astype('float32')
    output_data = np.load('coeff_data_2000.npy').astype('float32')
    adjoint_data = np.zeros(shape=(2000,8)).astype('float32') # placeholder

    # Define a simple fully connected model
    if args.adjoint_mode == 'augmented':
        model=coefficient_model_adjoint(input_data,output_data,adjoint_data)
    else:
        model=coefficient_model(input_data,output_data)

    # Train the model
    if args.mode == 'train':
        model.train_model()
    else:
        model.restore_model()

    if args.mode == 'optimize':
        # gradient-based Perform optimization
        from constraints import cons

        # Initialize optimizer
        num_pars = np.shape(input_data)[1]

        if args.constraint_options == 'shape_lift':
            lift_cons = True
            opt = surrogate_optimizer(model,num_pars,cons,lift_cons)
        else:
            opt = surrogate_optimizer(model,num_pars,cons)
        
        best_func_val, solution, best_opt = opt.optimize(30) # Optimize with 10 restarts
        
        # Visualize airfoil shape evolution
        for i in range(1,np.shape(best_opt)[0]):
            shape_return(best_opt[i],i)
   
    elif args.mode == 'rl_train':
        from rl_optimizers import rl_optimize
        from constraints import t_base
       
        # Create an RL based optimization
        env_params = {}
        env_params['num_params'] = np.shape(input_data)[1]
        env_params['num_obs'] = np.shape(output_data)[1]
        env_params['init_guess'] = np.asarray(t_base)
        env_params['model_type'] = 'regular'
        env_params['num_steps'] = 5 # 10 step iteration

        rl_opt = rl_optimize(env_params,args.adjoint_mode,50,env_params['num_steps']) # 20 iteration optimization, 10 steps
        rl_opt.train()

    elif args.mode == 'rl_optimize':
        from rl_optimizers import rl_optimize
        from constraints import t_base
       
        # Create an RL based optimization
        env_params = {}
        env_params['num_params'] = np.shape(input_data)[1]
        env_params['num_obs'] = np.shape(output_data)[1]
        env_params['init_guess'] = np.asarray(t_base,dtype='float32')
        env_params['model_type'] = 'regular'
        env_params['num_steps'] = 5

        rl_opt = rl_optimize(env_params,args.adjoint_mode,50,env_params['num_steps']) # 20 iteration optimization, 10 steps

        f = open('rl_checkpoint','r')
        checkpoint_path = f.readline()
        f.close()
        rl_opt.load_checkpoint(checkpoint_path)

        # Random restarts
        best_drag = 10.0
        for _ in range(30):
            path, coeff_path = rl_opt.optimize_shape()

            if coeff_path[-1][0] < best_drag:
                best_drag = coeff_path[-1][0]
                best_params = path[-1]
                best_path = path

        # Visualize airfoil shape evolution
        for i in range(0,len(path)):
            shape_return(best_path[i],i)

        # Print best coefficients etc
        print('Drag and lift coefficients',coeff_path[-1])
        print('Optimized shape parameters:',path[-1])