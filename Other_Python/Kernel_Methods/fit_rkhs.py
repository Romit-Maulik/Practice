# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:07:58 2020

@author: matth
"""

import pdb
import autograd.numpy as np
np.random.seed(1)
from autograd import value_and_grad 
import math
import matplotlib.pyplot as plt

from kernel_functions import kernels_dic

#%%
    
"""We define several useful functions"""
    
# Returns a random sample of the data, as a numpy array
def sample_selection(data, size):
    indices = np.arange(data.shape[0])
    sample_indices = np.sort(np.random.choice(indices, size, replace= False))
    
    return sample_indices

# This function creates a batch and associated sample
def batch_creation(data, batch_size, sample_proportion = 0.5):
    # If False, the whole data set is the mini-batch, otherwise either a 
    # percentage or explicit quantity.
    if batch_size == False:
        data_batch = data
        batch_indices = np.arange(data.shape[0])
    elif 0 < batch_size <= 1:
        batch_size = int(data.shape[0] * batch_size)
        batch_indices = sample_selection(data, batch_size)
        data_batch = data[batch_indices]
    else:
        batch_indices = sample_selection(data, batch_size)
        data_batch = data[batch_indices]
        

    # Sample from the mini-batch
    sample_size = math.ceil(data_batch.shape[0]*sample_proportion)
    sample_indices = sample_selection(data_batch, sample_size)
    
    return sample_indices, batch_indices


# Generate a prediction
def kernel_regression(X_train, X_test, Y_train, param, kernel_keyword = "RBF", regu_lambda = 0.000001):
    kernel = kernels_dic[kernel_keyword]
    k_matrix = kernel(X_train, X_train, param)
    k_matrix += regu_lambda * np.identity(k_matrix.shape[0])
    t_matrix = kernel(X_test, X_train, param) 
    prediction = np.matmul(t_matrix, np.matmul(np.linalg.inv(k_matrix), Y_train)) 
    return prediction

# Predicttimeseries

def kernel_extrapolate(X_train, X_test, Y_train, param, nsteps=1, kernel_keyword = "RBF", regu_lambda = 0.000001):
    kernel = kernels_dic[kernel_keyword]
    k_matrix = kernel(X_train, X_train, param)
    k_matrix += regu_lambda * np.identity(k_matrix.shape[0])
    A=np.matmul(np.linalg.inv(k_matrix), Y_train)
    arr = np.array([])
    
    X_test0=X_test
    isteps=int(nsteps/(X_test.shape[1]))+1
    for i in range(isteps):
        X_test1=X_test0
        t_matrix = kernel(X_test1, X_train, param) 
        prediction = np.matmul(t_matrix, A) 
        X_test0=prediction
        arr = np.append(arr, np.array(prediction[0,:]))
    arr=arr[0:nsteps]
    return arr


def replace_nan(array):
    for i in range(array.shape[0]):
        if math.isnan(array[i]) == True:
            print("Found nan value, replacing by 0")
            array[i] = 0
    return array

def sample_size_linear(iterations, range_tuple):
    
    return np.linspace(range_tuple[0], range_tuple[1], num = iterations)[::-1]
            
#%% Rho function

# The pi or selection matrix
def pi_matrix(sample_indices, dimension):
    pi = np.zeros(dimension)
    
    for i in range(dimension[0]):
        pi[i][sample_indices[i]] = 1
    
    return pi


def rho(parameters, matrix_data, Y_data, sample_indices,  kernel_keyword= "RBF", regu_lambda = 0.000001):
    kernel = kernels_dic[kernel_keyword]
    
    kernel_matrix = kernel(matrix_data, matrix_data, parameters)
    pi = pi_matrix(sample_indices, (sample_indices.shape[0], matrix_data.shape[0]))   
    
    sample_matrix = np.matmul(pi, np.matmul(kernel_matrix, np.transpose(pi)))
    
    Y_sample = Y_data[sample_indices]
    
    lambda_term = regu_lambda
    inverse_data = np.linalg.inv(kernel_matrix + lambda_term * np.identity(kernel_matrix.shape[0]))
    inverse_sample = np.linalg.inv(sample_matrix + lambda_term * np.identity(sample_matrix.shape[0]))
    top = np.tensordot(Y_sample, np.matmul(inverse_sample, Y_sample))
    
    bottom = np.tensordot(Y_data, np.matmul(inverse_data, Y_data))
    
    print(1-top/bottom)
    return 1 - top/bottom

def l2(parameters, matrix_data, Y, batch_indices, sample_indices, kernel_keyword = "RBF"):
    X_sample = matrix_data[sample_indices]
    Y_sample = Y[sample_indices]
    
    not_sample = [x for x in batch_indices not in sample_indices]
    X_not_sample = matrix_data[not_sample]
    Y_not_sample = Y[not_sample]
    prediction = kernel_regression(X_sample, X_not_sample, Y_sample, kernel_keyword)
    
    return np.dot(Y_not_sample - prediction, Y_not_sample- prediction)

#%% Grad functions

""" We define the gradient calculator function.Like rho, the gradient 
calculator function accesses the gradfunctions via a keyword"""

# Gradient calculator function. Returns an array
def grad_kernel(parameters, X_data, Y_data, sample_indices, kernel_keyword= "RBF", regu_lambda = 0.000001):
    grad_K = value_and_grad(rho)
    rho_value, gradient = grad_K(parameters, X_data, Y_data, sample_indices, kernel_keyword, regu_lambda = regu_lambda)
    return rho_value, gradient

#%% The class version of KF
class KernelFlowsP():
    
    def __init__(self, kernel_keyword, parameters):
        self.kernel_keyword = kernel_keyword
        self.parameters = np.copy(parameters)
        
        # Lists that keep track of the history of the algorithm
        self.rho_values = []
        self.grad_hist = []
        self.para_hist = []
        
        self.LR = 0.1
        self.beta = 0.9
        self.regu_lambda = 0.0001
    
    def get_hist(self):
        return self.param_hist, self.gradients, self.rho_values
        
    
    def save_model(self):
        np.save("param_hist", self.param_hist)
        np.save("gradients", self.gradients)
        np.save("rho_values", self.rho_values)
        
    def get_parameters(self):
        return self.parameters
    
    def set_LR(self, value):
        self.LR = value
        
    def set_beta(self, value):
        self.beta = value
    def set_train(self, train):
        self.train = train
        
    
    def fit(self, X, Y, iterations, batch_size = False, optimizer = "SGD", 
            learning_rate = 0.1, beta = 0.9, show_it = 100, regu_lambda = 0.000001, 
            adaptive_size = False, adaptive_range = (), proportion = 0.5, reduction_constant = 0.0):            

        self.set_LR(learning_rate)
        self.set_beta(beta)
        self.regu_lambda = regu_lambda
        
        self.X_train = np.copy(X)
        self.Y_train = np.copy(Y)
        momentum = np.zeros(self.parameters.shape, dtype = "float")
        
        # This is used for the adaptive sample decay
        rho_100 = []
        adaptive_mean = 0
        adaptive_counter = 0
        
        if adaptive_size == False or adaptive_size == "Dynamic":
            sample_size = proportion
        elif adaptive_size == "Linear":
            sample_size_array = sample_size_linear(iterations, adaptive_range) 
        else:
            print("Sample size not recognized")
            
        for i in range(iterations):
            if i % show_it == 0:
                print("parameters ", self.parameters)
            
            if adaptive_size == "Linear":
                sample_size = sample_size_array[i]
                
            elif adaptive_size == "Dynamic" and adaptive_counter == 100:
                if adaptive_mean != 0:
                    change = np.mean(rho_100) - adaptive_mean 
                else:
                    change = 0
                adaptive_mean = np.mean(rho_100)
                rho_100 = []
                sample_size += change - reduction_constant
                adaptive_counter= 0
                
            # Create a batch and a sample
            sample_indices, batch_indices = batch_creation(X, batch_size, sample_proportion = sample_size)
            X_data = X[batch_indices]
            Y_data = Y[batch_indices]
            

                
            # Changes parameters according to SGD rules
            if optimizer == "SGD":
                rho, grad_mu = grad_kernel(self.parameters, X_data, Y_data, 
                                           sample_indices, self.kernel_keyword, regu_lambda = regu_lambda)
                if  rho > 1 or rho < 0:
                    print("Warning, rho outside [0,1]: ", rho)
                else:
                    self.parameters -= learning_rate * grad_mu
                    
            
            # Changes parameters according to Nesterov Momentum rules     
            elif optimizer == "Nesterov":
                rho, grad_mu = grad_kernel(self.parameters - learning_rate * beta * momentum, 
                                               X_data, Y_data, sample_indices, self.kernel_keyword, regu_lambda = regu_lambda)
                if  rho > 1 or rho < 0:
                    print("Warning, rho outside [0,1]: ", rho)
                else:
                    momentum = beta * momentum + grad_mu
                    self.parameters -= learning_rate * momentum
                
            else:
                print("Error optimizer, name not recognized")
            
            # Update history 
            self.para_hist.append(np.copy(self.parameters))
            self.rho_values.append(rho)
            self.grad_hist.append(np.copy(grad_mu))
            
            rho_100.append(rho)
            adaptive_counter +=1
                
            
        # Convert all the lists to np arrays
        self.para_hist = np.array(self.para_hist) 
        self.rho_values = np.array(self.rho_values)
        self.grad_hist = np.array(self.grad_hist)
                
        return self.parameters
    
    def predict(self,test, regu_lambda = 0.0000001):
         
        X_train = self.X_train
        Y_train = self.Y_train
        prediction = kernel_regression(X_train, test, Y_train, self.parameters, self.kernel_keyword, regu_lambda = regu_lambda) 

        return prediction

    def extrapolate(self,test, nsteps=1,regu_lambda = 0.000001):
         
        X_train = self.X_train
        Y_train = self.Y_train
        prediction = kernel_extrapolate(X_train, test, Y_train, self.parameters, nsteps,self.kernel_keyword, regu_lambda = regu_lambda) 

        return prediction

def fit_data_anl3(train_data,in_delay,out_delay,regu_lambda,noptsteps=100):
    lenX=len(train_data[0,:])
    num_modes = train_data.shape[0]

    # Some constants
    nparameters=24
    vregu_lambda=regu_lambda*np.ones((num_modes,))

    # Get scaling factor    
    normalize=np.amax(train_data[:,:])

    # Prepare training data
    X=np.zeros((1+lenX-(in_delay+out_delay),in_delay*num_modes))
    Y=np.zeros((1+lenX-(in_delay+out_delay),out_delay*num_modes))
    for i in range(1+lenX-(in_delay+out_delay)):
        for mode in range(train_data.shape[0]):
              X[i,(mode*in_delay):(mode*in_delay+in_delay)]=train_data[mode,i:(i+in_delay)]
              Y[i,(mode*out_delay):(mode*out_delay+out_delay)]=train_data[mode,(i+in_delay):(i+in_delay+out_delay)]

    # Normalize
    X=X/normalize
    Y=Y/normalize     
    
    # Fit data
    c=np.zeros(nparameters)+1
    mu_1 = c
    kerneltype="anl3"
    K = KernelFlowsP(kerneltype, mu_1)
    mu_pred = K.fit(X, Y, noptsteps, optimizer = "Nesterov",  batch_size = 100, show_it = 500, regu_lambda=regu_lambda)
    mu_1=mu_pred
    c=mu_1

    kernel = kernels_dic[kerneltype]
    k_matrix = kernel(X, X, mu_1)
    k_matrix += regu_lambda * np.identity(k_matrix.shape[0])
    A=np.matmul(np.linalg.inv(k_matrix), Y)

    return k_matrix, A, mu_1, normalize, X

def test_fit_anl3(test_data,train_X,in_delay,out_delay,k_matrix,A,param,normalize):

    kerneltype = "anl3"
    kernel = kernels_dic[kerneltype]

    num_pred = test_data.shape[-1]
    num_modes = test_data.shape[0]


    truth_list = []; prediction_list = []
    for i in range(num_pred-in_delay):

        input_data = test_data[:,i:i+in_delay].reshape(1,-1)/normalize
        t_matrix = kernel(input_data, train_X, param) 
        prediction = np.matmul(t_matrix, A).reshape(num_modes,-1)*normalize
        truth = test_data[:,i+in_delay:i+in_delay+out_delay]

        truth_list.append(truth)
        prediction_list.append(prediction)

    return truth_list, prediction_list



if __name__ == '__main__':
    from time import time
    train_data = np.load('./Total_train_data.npy').T[:5,:1000] # Should be modes x timesteps
    test_data = np.load('./Total_test_data.npy').T[:5,:400]

    # Set delay
    in_delay = 42
    out_delay = 7
    regu_lambda = 5.0
    noptsteps = 100

    # Fit on training data
    start_time = time()
    k_matrix, A, param, normalize, train_X = fit_data_anl3(train_data,in_delay,out_delay,regu_lambda,noptsteps)
    end_time = time()

    print('Time taken for fit:',end_time-start_time)

    # Predict and get error on training data
    true_train, predicted_train = test_fit_anl3(train_data, train_X, in_delay, out_delay, k_matrix, A, param, normalize)
    np.save('RKHS_Train_Prediction.npy',predicted_train)
    

    # Predict testing data
    true_test, predicted_test = test_fit_anl3(test_data, train_X, in_delay, out_delay, k_matrix, A, param, normalize)
    np.save('RKHS_Test_Prediction.npy',predicted_test)

    # Visualize the modal predictions for a particular prediction
    pred_num = 10
    for mode_num in range(3):
        fig, ax = plt.subplots(nrows=1,ncols=2)
        ax[0].plot(predicted_test[pred_num][mode_num,:],label='Test predicted')
        ax[0].plot(true_test[pred_num][mode_num,:],label='Test true')
        ax[0].legend()

        ax[1].plot(predicted_train[pred_num][mode_num,:],label='Train predicted')
        ax[1].plot(true_train[pred_num][mode_num,:],label='Train true')
        ax[1].legend()
        plt.tight_layout()
        plt.show()