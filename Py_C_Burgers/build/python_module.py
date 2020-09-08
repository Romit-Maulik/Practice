print("From python: Within python module")

import os,sys
HERE = os.getcwd()
sys.path.insert(0,HERE)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

data_array = np.zeros(shape=(200,258))
x = np.arange(start=0,stop=2.0*np.pi,step=2.0*np.pi/256)
iternum = 0

def collection_func(input_array):
    global data_array,iternum
    data_array[iternum,:] = input_array[:]
    iternum+=1
    return None

def analyses_func(placeholder):

    global data_array, x
    
    plt.figure()
    for i in range(10,200,40):
        plt.plot(x,data_array[i,1:-1],label='Timestep '+str(i))
    plt.legend()
    plt.xlabel('x')
    plt.xlabel('u')
    plt.title('Field evolution')
    plt.show()

    # Perform an SVD
    data_array = data_array[:,1:-1]
    print('Performing SVD')
    u,s,v = np.linalg.svd(data_array,full_matrices=False)

    # Plot SVD eigenvectors
    plt.figure()
    plt.plot(x, v[0,:],label='Mode 0')
    plt.plot(x, v[1,:],label='Mode 1')
    plt.plot(x, v[2,:],label='Mode 2')
    plt.legend()
    plt.title('SVD Eigenvectors')
    plt.xlabel('x')
    plt.xlabel('u')
    plt.show()

    np.save('eigenvectors.npy',v[0:3,:].T)

    return v[0:3,:].T

if __name__ == '__main__':
    pass