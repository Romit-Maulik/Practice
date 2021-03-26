import numpy as np
import scipy.sparse as sparse
from scipy.sparse import linalg
import pandas as pd

# global variables
#This will change the initial condition used. Currently it starts from the first# value
shift_k = 0

approx_res_size = 5000


model_params = {'tau': 0.25,
                'nstep': 1000,
                'N': 12,
                'd': 22}

res_params = {'radius':0.1,
             'degree': 3,
             'sigma': 0.5,
             'train_length': 400,
             'spin_off': 50,
             'N': int(np.floor(approx_res_size/model_params['N']) * model_params['N']),
             'num_inputs': model_params['N'],
             'predict_length': 300,
             'beta': 0.0001
              }

# The ESN functions for training
def generate_reservoir(size,radius,degree):
    sparsity = degree/float(size);
    A = sparse.rand(size,size,density=sparsity).todense()
    vals = np.linalg.eigvals(A)
    e = np.max(np.abs(vals))
    A = (A/e) * radius
    return A

def reservoir_layer(A, Win, input, res_params):
    states = np.zeros((res_params['N'],res_params['train_length']))
    for i in range(res_params['train_length']-1):
        states[:,i+1] = np.tanh(np.dot(A,states[:,i]) + np.dot(Win,input[:,i]))
    return states

def reservoir_layer_reset(A, Win, input, res_params):
    states = np.zeros((res_params['N'],res_params['spin_off']))
    for i in range(res_params['spin_off']-1):
        states[:,i+1] = np.tanh(np.dot(A,states[:,i]) + np.dot(Win,input[:,i]))
    x_reset = states[:,-1]
    return x_reset

def train_reservoir(res_params, data):
    A = generate_reservoir(res_params['N'], res_params['radius'], res_params['degree'])
    q = int(res_params['N']/res_params['num_inputs'])
    Win = np.zeros((res_params['N'],res_params['num_inputs']))
    for i in range(res_params['num_inputs']):
        np.random.seed(seed=i)
        Win[i*q: (i+1)*q,i] = res_params['sigma'] * (-1 + 2 * np.random.rand(1,q)[0])
        
    states = reservoir_layer(A, Win, data, res_params)
    Wout = train(res_params, states, data)
    x = states[:,-1]
    return x, Wout, A, Win

def train(res_params,states,data):
    beta = res_params['beta']
    idenmat = beta * sparse.identity(res_params['N'])
    states2 = states.copy()
    for j in range(2,np.shape(states2)[0]-2):
        if (np.mod(j,2)==0):
            states2[j,:] = (states[j-1,:]*states[j-2,:]).copy()
    U = np.dot(states2,states2.transpose()) + idenmat
    Uinv = np.linalg.inv(U)
    Wout = np.dot(Uinv,np.dot(states2,data.transpose()))
    return Wout.transpose()

def predict(A, Win, res_params, x, Wout):
    output = np.zeros((res_params['num_inputs'],res_params['predict_length']))
    for i in range(res_params['predict_length']):
        x_aug = x.copy()
        for j in range(2,np.shape(x_aug)[0]-2):
            if (np.mod(j,2)==0):
                x_aug[j] = (x[j-1]*x[j-2]).copy()
        out = np.squeeze(np.asarray(np.dot(Wout,x_aug)))
        output[:,i] = out
        x1 = np.tanh(np.dot(A,x) + np.dot(Win,out))
        x = np.squeeze(np.asarray(x1))
    return output, x


if __name__ == '__main__':
    print('Echo state network file')
    '''
    Example of use
    # data=np.load('data.npy') # rows are state, columns are samples in time
    # Train reservoir
    x,Wout,A,Win = train_reservoir(res_params,data[:,shift_k:shift_k+res_params['train_length']])
    x_reset=reservoir_layer_reset(A, Win, data[:,shift_k:shift_k+res_params['spin_off']], res_params)
    # Prediction
    output, _ = predict(A, Win,res_params,x_reset,Wout)
    np.save('prediction_output',output)
    '''

