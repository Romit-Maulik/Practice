import numpy as np
import tensorflow as tf

np.random.seed(10)
tf.random.set_seed(10)

import talos

import fileinput
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import joblib

# Build neural network
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.regularizers import l1

input_data = np.load(add+'All_input_data.npy')
output_data = np.load(add+'All_output_data.npy')

num_data_points = np.shape(input_data)[0]
num_inputs = np.shape(input_data)[1]
num_outputs = np.shape(output_data)[1]

def new_r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred), axis=0)
    SS_tot = K.sum(K.square(y_true - K.mean(y_true, axis=0)), axis=0)
    output_scores =  1 - SS_res / (SS_tot + K.epsilon())
    r2 = K.mean(output_scores)
    return r2


def minimal(input_train_data,output_train_data,input_valid_data,output_valid_data):

    p = {'activation':['relu', 'elu'],
         'optimizer': ['Nadam', 'Adam'],
         'num_neurons': [10,30,50,60,70,90,110],
         'lrate': [0.0001,0.0005,0.001, 0.0015, 0.002],
         'batch_size': [20,30,40]
        }

    def nn_model(input_train_data,output_train_data,input_valid_data,output_valid_data,params):
        # Define model architecture here
        field_input = Input(shape=(num_inputs,),name='inputs')
        
        hidden_layer_1 = Dense(params['num_neurons'],activation=params['activation'])(field_input)
        hidden_layer_2 = Dense(params['num_neurons'],activation=params['activation'])(hidden_layer_1)
        hidden_layer_3 = Dense(params['num_neurons'],activation=params['activation'])(hidden_layer_2)
        hidden_layer_4 = Dense(params['num_neurons'],activation=params['activation'])(hidden_layer_3)
        hidden_layer_5 = Dense(params['num_neurons'],activation=params['activation'])(hidden_layer_4)

        outputs = Dense(num_outputs,name='outputs')(hidden_layer_5)

        model = Model(inputs=[field_input],outputs=[outputs])   

        model.compile(optimizer=params['optimizer'],
                  loss={'outputs': 'mean_squared_error'}, loss_weights=[1.0], metrics=[new_r2], learning_rate=params['lrate'])
        
        # model.summary()
        # Optimization
        # weights_filepath = 'best_weights.h5'
        # checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
        callbacks_list = [checkpoint,earlystopping]

        out = model.fit(input_train_data, output_train_data,
                         batch_size=params['batch_size'],
                         epochs=2000,callbacks=callbacks_list,
                         validation_data=(input_valid_data, output_valid_data),
                         verbose=1)

        return out, model


    scan_object = talos.Scan(input_data, output_data, model=nn_model, params=p, experiment_name='nn_search', fraction_limit=0.1)

if __name__ == '__main__':

    idx = np.arange(num_data_points)
    np.random.shuffle(idx)
    train_frac = int(0.8*num_data_points)

    input_train_data = input_data[idx[:train_frac]]
    output_train_data = output_data[idx[:train_frac]]

    input_valid_data = input_data[idx[train_frac:]]
    output_valid_data = output_data[idx[train_frac:]]
    
    # Preprocessing
    preproc_input = Pipeline([('stdscaler', StandardScaler()),('minmaxscaler', MinMaxScaler())])
    input_train_data = preproc_input.fit_transform(input_train_data)
    input_valid_data = preproc_input.transform(input_valid_data)
    scaler_filename = add+"ip_scaler.save"
    joblib.dump(preproc_input, scaler_filename)

    preproc_output = Pipeline([('stdscaler', StandardScaler()),('minmaxscaler', MinMaxScaler())])
    output_train_data = preproc_output.fit_transform(output_train_data)
    output_valid_data = preproc_output.fit_transform(output_valid_data)
    scaler_filename = add+"op_scaler.save"
    joblib.dump(preproc_output, scaler_filename)

    minimal(input_train_data,output_train_data,input_valid_data,output_valid_data)