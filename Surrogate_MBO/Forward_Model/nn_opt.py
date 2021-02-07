import argparse
import tensorflow as tf
tf.random.set_seed(10)

import numpy as np
np.random.seed(15)
import matplotlib.pyplot as plt

from surrogate_models import coefficient_model

if __name__ == '__main__':

    '''
    Usage: python nn_opt.py [train/restore]
    '''
    parser = argparse.ArgumentParser(description='Adjoint augmented surrogate model based optimization')
    parser.add_argument('mode', metavar='mode', type=str, help='[train/restore]') #Train a network or use a trained network for minimize/rl_minimize
    args = parser.parse_args()

    # Load dataset
    input_data = np.load('DOE_2000.npy').astype('float32')
    output_data = np.load('coeff_data_2000.npy').astype('float32')

    # Define a simple fully connected model
    model=coefficient_model(input_data,output_data)

    # Train the model
    if args.mode == 'train':
        model.train_model()
        model.restore_model()
    else:
        model.restore_model()

    # Predict
    inputs = np.asarray([0.1009, 0.3306, 0.6281, 0.1494, -0.1627, -0.6344, -0.5927, 0.0421]).astype('float32')
    inputs = inputs.reshape(1,8)
    pred = model.predict(inputs)
    print(pred)