import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as Kback
from tensorflow.keras.layers import Input, Dense, Lambda, Add, LSTM
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from Parameters import num_epochs, num_neurons, K, M, tsteps
from Compression import nl_reconstruct
from Parameters import tsteps, dt, seq_num

def create_slfn_model(data):
    '''
    data - time-series array : shape - timesteps x state length
    '''
    # Dataset prep
    net_inputs = data[0:-1,:]
    net_outputs = data[1:,:]
    
    # SLFN architecture
    input_t = Input(shape=(np.shape(net_inputs)[1],))
    l1 = Dense(num_neurons, activation='tanh')(input_t)
    output = Dense(np.shape(net_outputs)[1], activation='linear')(l1)
    
    # Optimizer
    my_adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # Some callbacks
    filepath = "best_weights_slfn.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
    callbacks_list = [checkpoint]
    
    # Compilation
    model = models.Model(inputs=input_t,outputs=output)
    model.compile(optimizer=my_adam,loss='mean_squared_error',metrics=[coeff_determination])
    
    # Training
    train_history = model.fit(net_inputs, net_outputs, epochs=num_epochs, batch_size=32, callbacks=callbacks_list)
    model.load_weights(filepath)

    return model, train_history

def check_apriori_performance_slfn(data,model):
    input_seq = np.zeros(shape=(1,np.shape(data)[1]))
    input_seq[0,:] = data[0,:]

    # Need to make batches of input sequences and 1 output
    total_size = np.shape(data)[0]
    apriori_tracker = np.zeros_like(data)
    apriori_tracker[0,:] = input_seq[0,:]
    
    for t in range(seq_num,total_size):
        output = model.predict(input_seq)
        apriori_tracker[t,:] = output[0,:]
        input_seq[0,:] = output[0,:]

    return np.transpose(apriori_tracker)

def create_lstm_model(data,mode='valid'):
    '''
    data - time-series array : shape - timesteps x state length
    '''
    # Need to make batches of input sequences and 1 output
    total_size = np.shape(data)[0]-seq_num
    input_seq = np.zeros(shape=(total_size,seq_num,np.shape(data)[1]))
    output_seq = np.zeros(shape=(total_size,np.shape(data)[1]))

    for t in range(total_size):
        input_seq[t,:,:] = data[None,t:t+seq_num,:]
        output_seq[t,:] = data[t+seq_num,:]

    idx = np.arange(total_size)
    np.random.shuffle(idx)
    
    input_seq = input_seq[idx,:,:]
    output_seq = output_seq[idx,:]
    
    # Model architecture
    model = models.Sequential()
    model.add(LSTM(num_neurons,input_shape=(seq_num, np.shape(data)[1])))  # returns a sequence of vectors of dimension 32
    model.add(Dense(np.shape(data)[1], activation='linear'))

    # design network
    my_adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    filepath = "best_weights_lstm.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
    callbacks_list = [checkpoint]
    
    # fit network
    model.compile(optimizer=my_adam,loss='mean_squared_error',metrics=[coeff_determination])

    if mode == 'train':
        train_history = model.fit(input_seq, output_seq, epochs=num_epochs, batch_size=16, validation_split=0.2, callbacks=callbacks_list)#validation_split = 0.1
        np.save('Train_Loss.npy',train_history.history['loss'])
        np.save('Val_Loss.npy',train_history.history['val_loss'])
        model.load_weights(filepath)
        return model, train_history
    else:
        model.load_weights(filepath)
        return model, None


def check_apriori_performance_lstm(data,model):
    input_seq = np.zeros(shape=(1,seq_num,np.shape(data)[1]))
    input_seq[0,:,:] = data[None,0:seq_num,:]

    # Need to make batches of input sequences and 1 output
    total_size = np.shape(data)[0]
    apriori_tracker = np.zeros_like(data)
    apriori_tracker[0:seq_num,:] = input_seq[0,:,:]
    
    for t in range(seq_num,total_size):
        output = model.predict(input_seq)
        input_seq[0,:-1,:] = input_seq[0,1:,:]
        input_seq[0,-1,:] = output[0,:]

        apriori_tracker[t,:] = output[0,:]

    return np.transpose(apriori_tracker)

def coeff_determination(y_pred, y_true): #Order of function inputs is important here        
    SS_res =  Kback.sum(Kback.square( y_true-y_pred )) 
    SS_tot = Kback.sum(Kback.square( y_true - Kback.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + Kback.epsilon()) )    

if __name__ == "__main__":
    print('This is the ML model definition file')