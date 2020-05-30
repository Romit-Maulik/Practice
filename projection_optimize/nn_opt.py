import argparse
import tensorflow as tf
tf.random.set_seed(10)

import numpy as np
np.random.seed(100)
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
    parser.add_argument('adjoint_mode', metavar='adjoint_mode', type=str, help='[regular/augmented/proj_regular/proj_augmented]') 
    args = parser.parse_args()

    # Load dataset
    input_data = np.load('doe_data.npy').astype('float32')
    output_data = np.load('coeff_data.npy').astype('float32')
    adjoint_data = np.zeros(shape=(170,8)).astype('float32') # placeholder
    field_data = np.load('pressure_data.npy').astype('float32')


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
        opt = surrogate_optimizer(model,num_pars,cons)
        best_func_val, solution, best_opt = opt.optimize(20) # Optimize with 10 restarts
        
        # Visualize airfoil shape evolution
        for i in range(1,np.shape(best_opt)[0]):
            shape_return(best_opt[i],i)
    
    elif args.mode == 'rl_train':
        from rl_optimizers import rl_optimize
       
        # Create an RL based optimization
        env_params = {}
        env_params['num_params'] = np.shape(input_data)[1]
        env_params['num_obs'] = np.shape(output_data)[1]
        env_params['init_guess'] = np.random.uniform(size=(np.shape(input_data)[1]))
        env_params['model_type'] = 'regular'

        rl_opt = rl_optimize(env_params,args.adjoint_mode,10) # 10 iteration optimization
        rl_opt.train()

    elif args.mode == 'rl_optimize':
        from rl_optimizers import rl_optimize
       
        # Create an RL based optimization
        env_params = {}
        env_params['num_params'] = np.shape(input_data)[1]
        env_params['num_obs'] = np.shape(output_data)[1]
        env_params['init_guess'] = np.random.uniform(size=(np.shape(input_data)[1]))
        env_params['model_type'] = 'regular'

        rl_opt = rl_optimize(env_params,args.adjoint_mode,20) # 10 iteration optimization

        f = open('rl_checkpoint','r')
        checkpoint_path = f.readline()
        f.close()
        rl_opt.load_checkpoint(checkpoint_path)
        path, coeff_path = rl_opt.optimize_shape()

        # Visualize airfoil shape evolution
        for i in range(1,len(path)):
            shape_return(path[i],i)