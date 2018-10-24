import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import os


def make_training_data():
    n_samples = 500    #Number of training data samples
    n_inputs = 2        #Number of input parameters
    n_outputs = 3       #Number of output parameters

    #Define arrays for storing these
    input_data = np.zeros(shape=(n_samples, n_inputs), dtype='double')
    output_data = np.zeros(shape=(n_samples, n_outputs), dtype='double')

    #Populate arrays
    np.random.seed(1)
    for i in range(n_samples):
        x = np.random.uniform(low=0.0, high=2.0 * np.pi)
        y = np.random.uniform(low=0.0, high=2.0 * np.pi)

        input_data[i, 0] = x
        input_data[i, 1] = y

        output_data[i, 0] = np.sin(x)*np.sin(y)
        output_data[i, 1] = np.sin(x)+np.cos(y)
        output_data[i, 2] = np.sin(-x-y)

    return n_samples, n_inputs, n_outputs, input_data, output_data


def plot_data(inputs,outputs):

    fig = plt.figure()
    ax = fig.add_subplot(311, projection='3d')
    ax.plot_trisurf(inputs[:,0],inputs[:,1],outputs[:,0],cmap=cm.jet, linewidth=0.2)
    ax.set_title('Function 1')
    ax.grid(False)
    ax.axis('off')

    ax = fig.add_subplot(312, projection='3d')
    ax.plot_trisurf(inputs[:,0],inputs[:,1],outputs[:,1],cmap=cm.jet, linewidth=0.2)
    ax.set_title('Function 2')
    ax.grid(False)
    ax.axis('off')


    ax = fig.add_subplot(313, projection='3d')
    ax.plot_trisurf(inputs[:,0],inputs[:,1],outputs[:,2],cmap=cm.jet, linewidth=0.2)
    ax.set_title('Function 3')
    ax.grid(False)
    ax.axis('off')

    plt.legend()
    plt.show()


    plt.figure()
    f1_true = outputs[:, 0].flatten()
    f2_true = outputs[:, 1].flatten()
    f3_true = outputs[:, 2].flatten()
    plt.hist(f1_true, bins=16, label=r'Function 1', histtype='step')  # arguments are passed to np.histogram
    plt.hist(f2_true, bins=16, label=r'Function 2', histtype='step')  # arguments are passed to np.histogram
    plt.hist(f3_true, bins=16, label=r'Function 3', histtype='step')  # arguments are passed to np.histogram
    plt.legend()
    plt.show()

if __name__ == "__main__":
    n_samples, n_inputs, n_outputs, input_data, output_data = make_training_data()
    plot_data(input_data, output_data)