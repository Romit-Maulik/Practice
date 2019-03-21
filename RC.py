import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random


def reservoir_computing():
    n_time_steps = 100
    input_signal_dim = 1    #N_u
    output_signal_dim = 1   #N_y
    n_states = 100         #N_x - vector of reservoir neurons
    leak_rate = 0.1
    np.random.seed(10)

    input_signal = np.zeros((n_time_steps+1,1+input_signal_dim),dtype='double')
    targets_signal = np.zeros((n_time_steps+1, output_signal_dim), dtype='double')

    '''
    Make some time series data - sine + cosine wave
    Concatenate input signal with 1
    '''
    for i in range(n_time_steps+1):
        input_signal[i, 0] = 1.0
        input_signal[i, 1] = float(i / n_time_steps)
        targets_signal[i,0] = np.sin(2.0*np.pi*i/n_time_steps) #+ np.cos(2.0*np.pi*(i-2)/n_time_steps)

    '''
    The first input signal and target signal will not be used for training due to recurrent nature of RC
    '''

    #Define input weight matrices - add sparsity
    #W_in = np.random.uniform(low=-1.0,high=1.0,size=(1+input_signal_dim,n_states))
    W_in = random(m=1+input_signal_dim,n=n_states,density=0.01).toarray()

    #Define recurrent weight matrix
    #W_rec = random(m=n_states, n=n_states, density=0.01).toarray()
    W_rec = np.random.uniform(low=-1.0,high=1.0,size=(n_states,n_states))

    #Define backprojection weight matrix - unused
    W_bac = np.random.uniform(low=-1.0,high=1.0,size=(output_signal_dim,n_states))

    '''
    Input and recurrent weight matrix utilized for calculation of state vector
    Needs current input signal and previous state vector
    '''

    '''
    Calculating hidden states for each sample
    '''
    hidden_states = np.zeros((n_time_steps+1,n_states),dtype='double')
    hidden_states_temp = np.zeros((n_time_steps + 1, n_states), dtype='double')

    #Fixing first entry
    hidden_states[0, :] = np.tanh(np.matmul(input_signal[0, :], W_in))
    hidden_states[0, :] = leak_rate * hidden_states[0, :]

    for i in range(1,n_time_steps+1):
        hidden_states_temp[i, :] = np.tanh(
            np.matmul(input_signal[i, :], W_in) + np.matmul(hidden_states[i - 1, :], W_rec) + np.matmul(targets_signal[i - 1, :], W_bac))

        hidden_states[i, :] = (1.0-leak_rate)*hidden_states[i-1,:] + leak_rate*hidden_states_temp[i,:]

    '''
    Final fixing of dimensions - 0:n_time_steps-1
    '''
    hidden_states = hidden_states[1:, ]
    input_signal = input_signal[1:, ]
    targets_signal = targets_signal[1:, ]


    '''
    Verifying feed-forward dimensions
    '''
    concatenated_feed = np.concatenate([input_signal,hidden_states],axis=1)

    '''
    Least-squares regression to find optimal W_op
    '''
    beta = 1.0
    W_op = np.matmul(np.linalg.pinv(concatenated_feed),targets_signal)
    predicted_signal = np.matmul(concatenated_feed, W_op)

    plt.figure()
    plt.plot(targets_signal,label='targets')
    plt.plot(predicted_signal,label='prediction')
    plt.legend()
    plt.show()

    '''
    Prediction using time-series
    Unvalidated    
    '''
    plt.figure()
    i = 0
    signal = np.zeros(shape=(1, 1 + input_signal_dim))
    signal[0, 0] = 1.0
    signal[0, 1] = float(i / n_time_steps)

    predicted_signal = np.zeros(shape=(1, output_signal_dim))
    predicted_signal[0,0] = 0.0


    '''
    Fixing first hidden state
    '''

    hidden_states = np.zeros((2, n_states), dtype='double')
    hidden_states_temp = np.zeros((2, n_states), dtype='double')

    # Fixing first entry
    hidden_states[0, :] = np.tanh(np.matmul(signal[0, :], W_in))
    hidden_states[0, :] = leak_rate * hidden_states[0, :]

    for i in range(1,n_time_steps):
        '''
        New input signal
        '''
        signal[0, 1] = float(i / n_time_steps)

        '''
        Calculating temporary hidden state for this sample
        '''
        hidden_states_temp[1, :] = np.tanh(
            np.matmul(signal[0, :], W_in) + np.matmul(hidden_states[0, :], W_rec) + np.matmul(predicted_signal[0, :], W_bac))
        '''
        Updating
        '''
        hidden_states[1, :] = (1.0 - leak_rate) * hidden_states[0, :] + leak_rate * hidden_states_temp[1, :]

        '''
        Calculating prediction with optimal W_op
        '''
        concatenated_feed = np.concatenate([signal, hidden_states[1:,:]], axis=1)
        predicted_signal = np.matmul(concatenated_feed, W_op)

        #print(predicted_signal[0,0])

        hidden_states[0,:] = hidden_states[1,:]
        plt.scatter(i,predicted_signal,color='blue')


    plt.plot(targets_signal,label='targets')
    plt.legend()
    plt.show()



reservoir_computing()