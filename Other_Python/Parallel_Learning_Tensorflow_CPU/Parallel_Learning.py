# run using: 'mpiexec -n 4 python Parallel_Learning.py' at command line
# Dependencies tensorflow=1.12.0, keras=2.2.4, mpi4py, numpy=1.14.3, python=3.6.8
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras import backend as K

from mpi4py import MPI

def coeff_determination(y_pred, y_true): #Order of function inputs is important here        
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def create_keras_model(features, targets, proc_num, phase, best_model=None):

    if proc_num==0:
        verb_num = 1
    else:
        verb_num = 0

    #Import parameters from dict
    n_inputs = np.shape(features)[1]

    # Layers start
    input_layer = Input(shape=(n_inputs,))

    # ANN for regression
    x = Dense(10, activation='relu', use_bias=True)(input_layer)

    op = Dense(1, activation='linear', use_bias=True)(x) # One target in this case

    custom_model = Model(inputs=input_layer, outputs=op)

    filepath = "best_model_"+str(proc_num)+".h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=verb_num, save_best_only=True, mode='min', save_weights_only=True)
    callbacks_list = [checkpoint]

    custom_model.compile(optimizer='adam', loss='mean_squared_error',metrics=[coeff_determination])

    if phase == 1:
        filepath_saved = "best_model_"+str(best_model)+".h5"
        custom_model.load_weights(filepath_saved)

    history_callback = custom_model.fit(features, targets, epochs=25, batch_size=512, verbose=verb_num,callbacks=callbacks_list)

    loss_history = history_callback.history["loss"]
    loss_history = np.array(loss_history)

    if phase == 1:
        current_loss = np.loadtxt("loss_history_"+str(proc_num)+".txt")
        total_loss_history = np.concatenate((current_loss,loss_history),axis=0)
        np.savetxt("loss_history_"+str(proc_num)+".txt", total_loss_history, delimiter=",")
    else:
        np.savetxt("loss_history_"+str(proc_num)+".txt", loss_history, delimiter=",")

    return None


def load_data(rank,nprocs):
    # Load master dataset
    global_data = np.load('Training_data.npy')

    # Determine stride of data to keep
    total_data = np.shape(global_data)[0]
    stride_start = int(rank/nprocs*total_data)
    stride_end = int((rank+1)/nprocs*total_data)
 
    training_data = global_data[stride_start:stride_end,:]

    # Select input features - the last column is assumed as the target
    features = training_data[:, 0:np.shape(training_data)[1] - 1]
    targets = np.reshape(training_data[:, np.shape(training_data)[1] - 1], newshape=(np.shape(training_data)[0], 1))

    return features, targets

if __name__ == "__main__":
    # MPI Fluff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # Seed based on process number (for random initializations)
    np.random.seed(rank)

    # Run concurrent trainings from random initializations on individual processes
    features, targets = load_data(rank,nprocs)  
    create_keras_model(features,targets,rank,0) #Not reloading any weights
    comm.Barrier()

    num_sweeps = 4
    synch_models = 'best'
    for i in range(num_sweeps):
        if synch_models == 'best':
            # Synchronize for the best model to restart search
            if rank == 0:
                best_proc = 0
                loss_val = 100.0
                for rval in range(nprocs):
                    proc_loss_val = np.loadtxt('loss_history_'+str(rval)+'.txt')[-1]
                    if proc_loss_val < loss_val:
                        best_proc = rval
                        loss_val = proc_loss_val
            else:
                best_proc = None
        elif synch_models == 'random':
        	# Randomly switch model weights
            if rank == 0:
                best_proc = np.random.randint(low=0, high=nprocs)
            else:
                best_proc = None

        best_proc = comm.bcast(best_proc,root=0)
        create_keras_model(features,targets,rank,1,best_proc)
        comm.Barrier()

    # Root communicates the best model process
    if rank == 0:
        best_proc = 0
        loss_val = 100.0
        for rval in range(nprocs):
            proc_loss_val = np.loadtxt('loss_history_'+str(rval)+'.txt')[-1]
            if proc_loss_val < loss_val:
                best_proc = rval
                loss_val = proc_loss_val

        print('Final best model in processor: ',best_proc)
