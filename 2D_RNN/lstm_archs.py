import os,sys 
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(dir_path)
sys.path.insert(0,dir_path)

import tensorflow as tf
tf.random.set_seed(10)
tf.keras.backend.set_floatx('float32')

from tensorflow.keras import Model
from detailed_layer import LSTM_grid_layer_v1, LSTM_grid_layer_v2, Original_LSTM_grid_layer
import numpy as np
np.random.seed(10)

from utils import coeff_determination
# preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Plotting
import matplotlib.pyplot as plt

def all_same(items):
    return all(x == items[0] for x in items)

#Build the model which does basic map of inputs to coefficients
class grid_lstm(Model):
    def __init__(self,data,input_idx,output_idx,input_horizon,output_horizon,resolution_list):
        super(grid_lstm, self).__init__()

        '''
        Data is a set of tuples
        input_horizon refers to the number of days that are inputs (integer)
        output_horizon refers to the number of days that are outputs (integer)
        resolution_list corresponds to the sampling rate for all the members of Data (integer list), daily is 1, hourly is 24 etc
        '''

        self.num_dof = len(data)
        self.data_list = data
        self.input_idx = input_idx
        self.output_idx = output_idx
        self.resolution_list = resolution_list
        self.input_horizon = input_horizon
        self.output_horizon = output_horizon

        self.prepoc_list = []
        self.state_len_list = []
        
        # Preprocess data
        for i in range(self.num_dof):

            preproc_pipeline = Pipeline([('minmax', MinMaxScaler(feature_range=(-1, 1)))])
            # preproc_pipeline = Pipeline([('standard', StandardScaler())])
            temp = preproc_pipeline.fit_transform(data[i]).copy()
            self.data_list[i] = temp
            self.prepoc_list.append(preproc_pipeline)

            # Set up the data for the LSTM
            data_tsteps = np.shape(data[i])[0]
            state_len = np.shape(data[i])[1]

            self.state_len_list.append(state_len)

        self.input_data_list = []
        self.output_data_list = []
        self.snum_list = []
        
        for i in input_idx:
            
            state_len = self.state_len_list[i]
            data = self.data_list[i]
            seq_num_ip = self.resolution_list[i]*self.input_horizon
            seq_num_op = self.resolution_list[i]*self.output_horizon

            total_size = np.shape(data)[0]-int(seq_num_ip+seq_num_op) # Limit of sampling (does not factor for resolution)

            input_seq = np.zeros(shape=(total_size,seq_num_ip,state_len))  #[samples,n_inputs,state_len]

            snum = 0
            for t in range(0,total_size,self.resolution_list[i]):
                input_seq[snum,:,:] = data[None,t:t+seq_num_ip,:]
                snum = snum + 1

            self.input_data_list.append(input_seq[:snum]) # Ad-hoc hack for correcting resolution
            self.snum_list.append(snum)

        for i in output_idx:
            
            state_len = self.state_len_list[i]
            data = self.data_list[i]

            seq_num_ip = self.resolution_list[i]*self.input_horizon
            seq_num_op = self.resolution_list[i]*self.output_horizon

            total_size = np.shape(data)[0]-int(seq_num_ip+seq_num_op) # Limit of sampling

            output_seq = np.zeros(shape=(total_size,seq_num_op,state_len))  #[samples,n_inputs,state_len]

            snum = 0
            for t in range(0,total_size,self.resolution_list[i]):
                output_seq[snum,:,:] = data[None,t+seq_num_ip:t+seq_num_ip+seq_num_op,:]
                snum = snum + 1

            self.output_data_list.append(output_seq[:snum])
            self.snum_list.append(snum)

        if not all_same(self.snum_list):
            print('Check input and output horizons for consistency')
            exit()


        # Shuffle datasets
        self.input_data_train_list = []
        self.input_data_valid_list = []

        self.output_data_train_list = []
        self.output_data_valid_list = []


        idx = np.arange(self.snum_list[0])
        self.num_train = int(0.8*self.snum_list[0])
        self.num_valid = self.snum_list[0] - self.num_train

        np.random.shuffle(idx)

        for i in range(len(self.input_data_list)):
            self.input_data_train_list.append(self.input_data_list[i][idx[:self.num_train]])
            self.input_data_valid_list.append(self.input_data_list[i][idx[self.num_train:]])

        for i in range(len(self.output_data_list)):
            self.output_data_train_list.append(self.output_data_list[i][idx[:self.num_train]])
            self.output_data_valid_list.append(self.output_data_list[i][idx[self.num_train:]])

        # Construct network
        self.make_architecture()

        # Minibatch information



        ####### This section for testing #######

        # print(self.input_data_train_list[0].shape)
        # print(self.input_data_train_list[1].shape)
        # print(self.output_data_train_list[0].shape)

        # state_len_list = [self.state_len_list[i] for i in input_idx]
        # Xtest = [tf.convert_to_tensor(self.input_data_train_list[0][0:3]),
        #         tf.convert_to_tensor(self.input_data_train_list[1][0:3]),
        #         tf.convert_to_tensor(self.input_data_train_list[2][0:3])]
        # test_layer = Original_LSTM_grid_layer(input_dim_list=state_len_list,seq_length=self.input_horizon)
        # hh, mm = test_layer(Xtest)
        # print(tf.shape(hh))
        # exit()
        
        # # Test feed-forward pass
        # Xtest = [tf.convert_to_tensor(self.input_data_train_list[0][0:3]),
        #         tf.convert_to_tensor(self.input_data_train_list[1][0:3]),
        #         tf.convert_to_tensor(self.input_data_train_list[2][0:3])]

        # # op_list = self.call(Xtest)
        # # print(op_list)

        # Ytrue = tf.convert_to_tensor(self.output_data_train_list[0][0:3],dtype=tf.float32)
        # print(self.get_loss(Xtest,Ytrue))


    def make_architecture(self):

        input_state_dims = [self.state_len_list[i] for i in self.input_idx]
        input_seq_lengths = [self.resolution_list[i]*self.input_horizon for i in self.input_idx]

        output_state_dims = [self.state_len_list[i] for i in self.output_idx]
        output_seq_lengths = [self.resolution_list[i]*self.output_horizon for i in self.output_idx]

        # Grid LSTM layer for encoding all inputs - constant and uniform temporal resolution
        self.encoder_layer = Original_LSTM_grid_layer(input_dim_list=input_state_dims,seq_length=self.input_horizon)

        self.repeater_list = []
        self.output_lstm_list = []
        self.output_layer_list = []
        for i in range(len(self.output_idx)):

            self.repeater_list.append(tf.keras.layers.RepeatVector(self.state_len_list[self.output_idx[i]]))
                    
            self.output_lstm_list.append(
                tf.keras.layers.LSTM(10,
                    return_sequences=True,
                    activation='tanh',
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=tf.keras.regularizers.L1(0.01)
                    )
                )
            
            self.output_layer_list.append(
                # tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(
                        self.state_len_list[self.output_idx[i]],
                        kernel_initializer='glorot_normal',
                        kernel_regularizer=tf.keras.regularizers.L1(0.01)
                        )
                    # )
                )

        self.train_op = tf.keras.optimizers.Adam(learning_rate=0.01)

    
    # Running the model
    @tf.function
    def call(self,X):
        '''
        X must be a list of tensors
        '''
        # Input encoder
        hh, mm = self.encoder_layer(X) # Returns two lists
        H_ = tf.concat(hh,axis=-1)
        M_ = tf.concat(mm,axis=-1)
        # encoded = tf.concat([H_,M_],axis=-1)

        # Output variables
        output_tensor_list = []
        for i in range(len(self.output_idx)):
        
            # Call intermediate LSTM
            H_temp = tf.expand_dims(H_,axis=1)
            H_temp = tf.repeat(H_temp,repeats=self.resolution_list[self.output_idx[i]]*self.output_horizon,axis=1)
            M_temp = tf.expand_dims(M_,axis=1)
            M_temp = tf.repeat(M_temp,repeats=self.resolution_list[self.output_idx[i]]*self.output_horizon,axis=1)


            # Retrieve layers
            lstm_ = self.output_lstm_list[i]
            output_layer_ = self.output_layer_list[i]

            lstm_input = tf.concat([H_temp,M_temp],axis=-1)

            H_temp = lstm_(lstm_input)
            out_ = output_layer_(H_temp)

            output_tensor_list.append(out_)

        return output_tensor_list

    
    # Regular MSE
    @tf.function
    def get_loss(self,X,Y):
        op=self.call(X)

        loss = tf.zeros(shape=(1,), dtype=tf.dtypes.float32, name=None)

        for i in range(len(self.output_idx)):
            loss = loss + tf.reduce_mean(tf.math.square(op[i]-Y[i]))
            # loss = loss + self.output_lstm_list[i].losses
            # loss = loss + self.output_layer_list[i].losses

        # loss = loss + self.encoder_layer.regularizer_loss()

        return loss

    # get gradients
    @tf.function
    def get_grad(self,X,Y):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(X,Y)
            g = tape.gradient(L, self.trainable_variables)
        return g
    
    # perform gradient descent
    @tf.function
    def network_learn(self,X,Y):
        g = self.get_grad(X,Y)
        self.train_op.apply_gradients(zip(g, self.trainable_variables))

    # Train the model
    def train_model(self, batch_size, num_epochs):
        plot_iter = 0
        stop_iter = 0
        patience = 20
        best_valid_loss = 1.0e16

        self.num_batches_train = int(self.num_train/batch_size)
        self.num_batches_valid = int(self.num_valid/batch_size)
        
        for i in range(num_epochs):
            # Training loss
            print('Training iteration:',i)
            
            train_loss = 0.0
            for batch in range(self.num_batches_train):

                input_batch = [self.input_data_train_list[i][batch*batch_size:(batch+1)*batch_size].astype('float32') for i in range(len(self.input_idx))]
                output_batch = [self.output_data_train_list[i][batch*batch_size:(batch+1)*batch_size].astype('float32') for i in range(len(self.output_idx))]

                # input_batch = tf.convert_to_tensor(input_batch)
                # output_batch = tf.convert_to_tensor(output_batch)

                train_loss = train_loss + self.get_loss(input_batch,output_batch).numpy()

                self.network_learn(input_batch,output_batch)

            train_loss = train_loss/self.num_batches_train
            print('Training loss:',train_loss)


            # Validation loss
            valid_loss = 0.0

            for batch in range(self.num_batches_valid):
                input_batch = [self.input_data_valid_list[i][batch*batch_size:(batch+1)*batch_size].astype('float32') for i in range(len(self.input_idx))]
                output_batch = [self.output_data_valid_list[i][batch*batch_size:(batch+1)*batch_size].astype('float32') for i in range(len(self.output_idx))]

                # input_batch = tf.convert_to_tensor(input_batch)
                # output_batch = tf.convert_to_tensor(output_batch)
            

                valid_loss = valid_loss + self.get_loss(input_batch,output_batch).numpy()

            valid_loss = valid_loss/self.num_batches_valid

            
            # Check early stopping criteria - if not satisfied reduce lr
            if valid_loss < best_valid_loss:
                
                print('Improved validation loss from:',best_valid_loss,' to:', valid_loss)
                
                best_valid_loss = valid_loss
                self.save_weights('./checkpoints/my_checkpoint')
                
                stop_iter = 0
            else:
                print('Validation loss (no improvement):',valid_loss)
                stop_iter = stop_iter + 1

            if stop_iter == patience:
                # Decay but don't stop
                self.train_op.learning_rate = self.train_op.learning_rate*0.5
                stop_iter = 0
                print('No improvement so halving learning rate')
                self.restore_model()

                if self.train_op.learning_rate < 1.0e-5:
                    break


    # Load weights
    def restore_model(self):
        self.load_weights(dir_path+'/checkpoints/my_checkpoint') # Load pretrained model

    # Do some testing
    def model_inference(self,test_data):
        '''
        Test data is a list of all the variables in test range

        '''
        if len(test_data) != self.num_dof:
            print('Test data is not compatible - numbers of variables are different')
            exit()

        self.data_list_test = []
        self.snum_list_test = []
        
        for i in range(self.num_dof):
            
            preproc_pipeline = self.prepoc_list[i]
            data = preproc_pipeline.transform(test_data[i])
            self.data_list_test.append(data)


        self.input_data_list_test = []
        for i in self.input_idx:

            data = self.data_list_test[i]

            state_len = self.state_len_list[i]
            seq_num_ip = self.resolution_list[i]*self.input_horizon
            seq_num_op = self.resolution_list[i]*self.output_horizon
            total_size = np.shape(data)[0]-int(seq_num_ip+seq_num_op) # Limit of sampling (does not factor for resolution)

            input_seq = np.zeros(shape=(total_size,seq_num_ip,state_len))  #[samples,n_inputs,state_len]

            snum = 0
            for t in range(0,total_size,self.resolution_list[i]):
                input_seq[snum,:,:] = data[None,t:t+seq_num_ip,:]
                snum = snum + 1

            self.input_data_list_test.append(input_seq[:snum]) # Ad-hoc hack for correcting resolution
            self.snum_list_test.append(snum)


        self.output_data_list_test = []
        for i in self.output_idx:
            
            state_len = self.state_len_list[i]
            data = self.data_list_test[i]

            seq_num_ip = self.resolution_list[i]*self.input_horizon
            seq_num_op = self.resolution_list[i]*self.output_horizon

            total_size = np.shape(data)[0]-int(seq_num_ip+seq_num_op) # Limit of sampling

            output_seq = np.zeros(shape=(total_size,seq_num_op,state_len))  #[samples,n_inputs,state_len]

            snum = 0
            for t in range(0,total_size,self.resolution_list[i]):
                output_seq[snum,:,:] = data[None,t+seq_num_ip:t+seq_num_ip+seq_num_op,:]
                snum = snum + 1

            self.output_data_list_test.append(output_seq[:snum])
            self.snum_list_test.append(snum)


        if not all_same(self.snum_list_test):
            print('Check input and output horizons for consistency in test data')
            exit()


        # Restore from checkpoint
        self.restore_model()

        # # Test feed-forward pass
        # Xtest = [tf.convert_to_tensor(self.input_data_train_list[0][0:3]),
        #         tf.convert_to_tensor(self.input_data_train_list[1][0:3]),
        #         tf.convert_to_tensor(self.input_data_train_list[2][0:3])]

        # # op_list = self.call(Xtest)
        # # print(op_list)

        # Make predictions
        # Inputs
        Xtest = [self.input_data_list_test[i] for i in range(len(self.input_idx))]
        # Output
        Ypred = self.call(Xtest)

        self.output_data_list_pred = []
        for i in range(len(self.output_idx)):
            self.output_data_list_pred.append(Ypred[i])

        # Rescale       
        for j in range(len(self.output_idx)):
            preproc_pipeline = self.prepoc_list[self.output_idx[j]]

            temp_data = self.output_data_list_test[j].reshape(snum*self.output_horizon,-1)
            temp_data = preproc_pipeline.inverse_transform(temp_data)
            self.output_data_list_test[j] = temp_data.reshape(snum,self.output_horizon,-1)

            temp_data = self.output_data_list_pred[j].numpy().reshape(snum*self.output_horizon,-1)
            temp_data = preproc_pipeline.inverse_transform(temp_data)
            self.output_data_list_pred[j] = temp_data.reshape(snum,self.output_horizon,-1)

        return self.output_data_list_test, self.output_data_list_pred


if __name__ == '__main__':
    print('Architecture file')