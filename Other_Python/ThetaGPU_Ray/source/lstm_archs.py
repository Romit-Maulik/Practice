import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(dir_path)

import tensorflow as tf
tf.random.set_seed(10)
# tf.keras.backend.set_floatx('float64')

from tensorflow.keras import Model
import numpy as np
np.random.seed(10)

from utils import coeff_determination
# preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Plotting
import matplotlib.pyplot as plt

# 3D Var
from scipy.optimize import minimize

#Build the model which does basic map of inputs to coefficients
class emulator(Model):
    def __init__(self,data,checkpoint_path,params,model_choice):
        super(emulator, self).__init__()

        self.num_obs = params[0]

        # Set up the data for the LSTM
        self.data_tsteps = np.shape(data)[0]
        self.state_len = np.shape(data)[1]

        if params[9] == 1:
            self.scaling = 1
            # Affine transformation
            self.scale_factor = np.max(np.abs(data))
            self.data = data/self.scale_factor

            self.trans_factor = np.abs(np.min(self.data)) + 1.0e-6
            self.data = self.data + self.trans_factor
        else:
            self.scaling = 0
            self.preproc_pipeline = Pipeline([('minmaxscaler', MinMaxScaler())])
            self.data = self.preproc_pipeline.fit_transform(data)

        # Need to make minibatches
        self.seq_num = params[1]
        self.seq_num_op = params[2]
        self.seq_num_gap = params[10]

        self.total_size = np.shape(data)[0]-int(self.seq_num_op+self.seq_num+self.seq_num_gap) # Limit of sampling

        input_seq = np.zeros(shape=(self.total_size,self.seq_num,self.state_len))  #[samples,n_inputs,state_len]
        output_seq = np.zeros(shape=(self.total_size,self.seq_num_op,self.state_len)) #[samples,n_outputs,state_len]

        snum = 0
        for t in range(0,self.total_size):
            input_seq[snum,:,:] = self.data[None,t:t+self.seq_num,:]
            output_seq[snum,:] = self.data[None,t+self.seq_num+self.seq_num_gap:t+self.seq_num+self.seq_num_gap+self.seq_num_op,:]        
            snum = snum + 1

        # Shuffle dataset
        idx = np.arange(snum)
        np.random.shuffle(idx)
        input_seq = input_seq[idx]
        output_seq = output_seq[idx]

        # Split into train and valid
        self.ntrain = int(params[3]*np.shape(input_seq)[0])
        self.nvalid = np.shape(input_seq)[0] - self.ntrain

        self.input_seq_train = input_seq[:self.ntrain]
        self.output_seq_train = output_seq[:self.ntrain]

        self.input_seq_valid = input_seq[self.ntrain:]
        self.output_seq_valid = output_seq[self.ntrain:]

        # Define architecture
        xavier=tf.keras.initializers.GlorotUniform()
        self.train_op = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model_choice = model_choice

        if self.model_choice == 'LSTM':
            self.l1=tf.keras.layers.LSTM(50,return_sequences=True,input_shape=(self.seq_num,self.state_len),activation='relu')
            self.l1_transform = tf.keras.layers.Dense(self.seq_num_op)
            self.l2=tf.keras.layers.LSTM(50,return_sequences=True,activation='relu')
            self.out = tf.keras.layers.Dense(self.state_len)

        elif self.model_choice == 'BLSTM':
            
            self.l1=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50,return_sequences=True,activation='relu'),input_shape=(self.seq_num,self.state_len))
            self.l1_transform = tf.keras.layers.Dense(self.seq_num_op)
            self.l2=tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50,return_sequences=True,activation='relu'))
            self.out = tf.keras.layers.Dense(self.state_len)            
        
        elif self.model_choice == 'LSTM_REPEAT':

            self.l1=tf.keras.layers.LSTM(50,input_shape=(self.seq_num,self.state_len),activation='relu')
            self.l2= tf.keras.layers.RepeatVector(self.seq_num_op)
            self.l3=tf.keras.layers.LSTM(50,return_sequences=True,activation='relu')       
            self.out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.state_len))

        elif self.model_choice == 'BLSTM_REPEAT':
            self.l1= tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50,activation='relu'),input_shape=(self.seq_num,self.state_len),)
            self.l2= tf.keras.layers.RepeatVector(self.seq_num_op)
            self.l3= tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50,return_sequences=True,activation='relu'))
            self.out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.state_len))

        elif self.model_choice == 'LSTM_ATT':

            self.l1=tf.keras.layers.LSTM(50,input_shape=(self.seq_num,self.state_len),activation='relu')
            self.l2= tf.keras.layers.RepeatVector(self.seq_num_op)
            self.l3_q=tf.keras.layers.LSTM(50,return_sequences=True,activation='relu')       
            self.l3_v=tf.keras.layers.LSTM(50,return_sequences=True,activation='relu')       
            self.out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.state_len))

        elif self.model_choice == 'LSTM_PROG':
            if self.seq_num_gap != 0:
                print('Progressive architecture only possible with zero output gap')
                exit()

            # Split prediction into three connected components
            self.split_num_1 = int(self.seq_num_op//3)
            self.split_num_2 = int(self.seq_num_op//3)
            self.split_num_3 = self.seq_num_op - 2*self.split_num_1


            self.l1_1= tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50,activation='relu'),input_shape=(self.seq_num,self.state_len),name='LSTM1_1')
            self.l2_1= tf.keras.layers.RepeatVector(self.split_num_1,name='REPEAT_1')
            self.l3_1= tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50,return_sequences=True,activation='relu'),name='LSTM1_2')
            self.out_1 = tf.keras.layers.Dense(self.state_len,name='OP_1')

            self.l2_2= tf.keras.layers.RepeatVector(self.split_num_2,name='REPEAT_2')
            self.l3_2= tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50,return_sequences=True,activation='relu'),name='LSTM2_2')
            self.out_2 = tf.keras.layers.Dense(self.state_len,name='OP_2')

            self.l2_3= tf.keras.layers.RepeatVector(self.split_num_3,name='REPEAT_3')
            self.l3_3= tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50,return_sequences=True,activation='relu'),name='LSTM3_2')
            self.out_3 = tf.keras.layers.Dense(self.state_len,name='OP_3')

        # For future use of hyperparameters
        self.params = params

        # # Prioritize according to scaled singular values
        # self.singular_values = np.load('Singular_Values.npy')[:self.state_len]
        # self.singular_values = self.singular_values/self.singular_values[0]

        # self.singular_values[:] = 1.0

        # 3D VAR duration
        self.var_duration = params[4]

        # Some LSTM specifics
        self.num_train_epochs = params[5]
        self.checkpoint_path = checkpoint_path

    # Running the model
    def call(self,X):
        
        if self.model_choice == 'LSTM' or self.model_choice == 'BLSTM':
            h1 = self.l1(X)
            h2 = tf.transpose(h1,perm=[0,2,1])
            h3 = self.l1_transform(h2)
            h4 = tf.transpose(h3,perm=[0,2,1])
            h5 = self.l2(h4)
            out = self.out(h5)

            return out
        
        elif self.model_choice == 'LSTM_REPEAT' or self.model_choice == 'BLSTM_REPEAT':
            hh = self.l1(X)
            hh = self.l2(hh)
            hh = self.l3(hh)
            out = self.out(hh)

            return out

        elif self.model_choice == 'LSTM_ATT':
            hh = self.l1(X)
            hh = self.l2(hh)
            hh_q = self.l3_q(hh)
            hh_v = self.l3_v(hh)

            hh = tf.keras.layers.Attention()([hh_q,hh_v])
            out = self.out(hh)

            return out

        elif self.model_choice == 'LSTM_PROG':

            hh = self.l1_1(X)
            hh = self.l2_1(hh)
            hh = self.l3_1(hh)
            out_1 = self.out_1(hh)

            hh = self.l1_1(X)
            hh = self.l2_2(hh)
            hh = self.l3_2(hh)
            out_2 = self.out_2(hh)

            hh = self.l1_1(X)
            hh = self.l2_3(hh)
            hh = self.l3_3(hh)
            out_3 = self.out_3(hh)

            return out_1, out_2, out_3

    def call_inference(self,X):
        if self.model_choice != 'LSTM_PROG':
            out = self.call(X)
        else:
            out_1, out_2, out_3 = self.call(X)
            out = tf.concat([out_1,out_2,out_3],axis=1)

        return out

    def get_loss_valid(self,X,Y):
        if self.model_choice != 'LSTM_PROG':
            out = self.get_loss(X,Y)
        else:
            out_1, out_2, out_3 = self.get_loss(X,Y)
            out = out_1 + out_2 + out_3

        return out

    
    # Regular MSE
    def get_loss(self,X,Y):

        if self.model_choice != 'LSTM_PROG':

            op=self.call(X)

            if self.params[11] == 0:
                temp = tf.reduce_mean(tf.math.square(op-Y),axis=0)
                temp = tf.reduce_mean(temp,0)
                temp = tf.reduce_mean(temp)

            else:
                temp = tf.reduce_mean(tf.math.abs(op-Y)/(tf.math.abs(Y)+1.0e-8),axis=0)
                temp = tf.reduce_mean(temp,0)
                temp = tf.reduce_mean(temp)

            return temp

        else:

            op1, op2, op3 =self.call(X)

            Y1 = Y[:,:self.split_num_1]
            Y2 = Y[:,self.split_num_1:self.split_num_1+self.split_num_2]
            Y3 = Y[:,self.split_num_1+self.split_num_2:]

            if self.params[11] == 0:

                temp1 = tf.reduce_mean(tf.math.square(op1-Y1),axis=0)
                temp1 = tf.reduce_mean(temp1,0)
                temp1 = tf.reduce_mean(temp1)


                temp2 = tf.reduce_mean(tf.math.square(op2-Y2),axis=0)
                temp2 = tf.reduce_mean(temp2,0)
                temp2 = tf.reduce_mean(temp2)

                temp3 = tf.reduce_mean(tf.math.square(op3-Y3),axis=0)
                temp3 = tf.reduce_mean(temp3,0)
                temp3 = tf.reduce_mean(temp3)

            else:

                temp1 = tf.reduce_mean(tf.math.abs(op1-Y1)/(tf.math.abs(Y1)+1.0e-8),axis=0)
                temp1 = tf.reduce_mean(temp1,0)
                temp1 = tf.reduce_mean(temp1)


                temp2 = tf.reduce_mean(tf.math.abs(op2-Y2)/(tf.math.abs(Y2)+1.0e-8),axis=0)
                temp2 = tf.reduce_mean(temp2,0)
                temp2 = tf.reduce_mean(temp2)

                temp3 = tf.reduce_mean(tf.math.abs(op3-Y3)/(tf.math.abs(Y3)+1.0e-8),axis=0)
                temp3 = tf.reduce_mean(temp3,0)
                temp3 = tf.reduce_mean(temp3)

            return temp1, temp2, temp3


    # get gradients - regular
    def get_grad(self,X,Y):
        if self.model_choice != 'LSTM_PROG':
            
            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                L = self.get_loss(X,Y)
                g = tape.gradient(L, self.trainable_variables)
            return g
        
        else:
            
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(self.trainable_variables)
                L1, L2, L3 = self.get_loss(X,Y)

                g1_trainable_variables= self.l1_1.trainable_variables + \
                                        self.l2_1.trainable_variables + \
                                        self.l3_1.trainable_variables + \
                                        self.out_1.trainable_variables

                g2_trainable_variables= self.l1_1.trainable_variables + \
                                        self.l2_2.trainable_variables + \
                                        self.l3_2.trainable_variables + \
                                        self.out_2.trainable_variables

                g3_trainable_variables= self.l1_1.trainable_variables + \
                                        self.l2_3.trainable_variables + \
                                        self.l3_3.trainable_variables + \
                                        self.out_3.trainable_variables

                g1 = tape.gradient(L1,g1_trainable_variables)
                g2 = tape.gradient(L2,g2_trainable_variables)
                g3 = tape.gradient(L3,g3_trainable_variables)

            return g1, g2, g3, g1_trainable_variables, g2_trainable_variables, g3_trainable_variables

    
    # perform gradient descent - regular
    def network_learn(self,X,Y):
        if self.model_choice != 'LSTM_PROG':
            g = self.get_grad(X,Y)
            self.train_op.apply_gradients(zip(g, self.trainable_variables))
        else:
            g1, g2, g3, g1_trainable_variables, g2_trainable_variables, g3_trainable_variables = self.get_grad(X,Y)
            self.train_op.apply_gradients(zip(g1,g1_trainable_variables))
            self.train_op.apply_gradients(zip(g2,g2_trainable_variables))
            self.train_op.apply_gradients(zip(g3,g3_trainable_variables))

    # Train the model
    def train_model(self):
        plot_iter = 0
        stop_iter = 0
        patience = 10
        best_valid_loss = 999999.0 # Some large number 

        self.num_batches = 5
        self.train_batch_size = int(self.ntrain/self.num_batches)
        self.valid_batch_size = int((self.nvalid)/self.num_batches)
        
        for i in range(self.num_train_epochs):
            # Training loss
            print('Training iteration:',i)
            
            for batch in range(self.num_batches):
                input_batch = self.input_seq_train[batch*self.train_batch_size:(batch+1)*self.train_batch_size]
                output_batch = self.output_seq_train[batch*self.train_batch_size:(batch+1)*self.train_batch_size]
                self.network_learn(input_batch,output_batch)

            # Validation loss
            valid_loss = 0.0
            valid_r2 = 0.0

            for batch in range(self.num_batches):
                
                input_batch = self.input_seq_valid[batch*self.valid_batch_size:(batch+1)*self.valid_batch_size]
                output_batch = self.output_seq_valid[batch*self.valid_batch_size:(batch+1)*self.valid_batch_size]
                            
                valid_loss = valid_loss + self.get_loss_valid(input_batch,output_batch).numpy()
                predictions = self.call_inference(self.input_seq_valid)

                if self.params[11] == 0:
                    valid_r2 = valid_r2 + coeff_determination(predictions,self.output_seq_valid)

            valid_r2 = valid_r2/(batch+1)


            # Check early stopping criteria
            if valid_loss < best_valid_loss:
                
                print('Improved validation loss from:',best_valid_loss,' to:', valid_loss)
                print('Validation R2:',valid_r2)
                
                best_valid_loss = valid_loss

                self.save_weights(self.checkpoint_path+'./checkpoints/my_checkpoint')
                
                stop_iter = 0
            else:
                print('Validation loss (no improvement):',valid_loss)

                if self.params[11] == 0:
                    print('Validation R2:',valid_r2)
                stop_iter = stop_iter + 1

            if stop_iter == patience:
                break

    # Load weights
    def restore_model(self):
        try:
            self.load_weights(self.checkpoint_path+'./checkpoints/my_checkpoint') # Load pretrained model
        except:
            print('Cannot find trained model in path specified.')
            exit()

    def regular_inference(self,test_data):
        # Restore from checkpoint
        self.restore_model()

        # Scale testing data
        if self.scaling == 1:
            test_data = test_data/self.scale_factor
            test_data = test_data + self.trans_factor
        else:
            test_data = self.preproc_pipeline.transform(test_data)

        # Test data has to be scaled already
        test_total_size = np.shape(test_data)[0]-int(self.seq_num_op+self.seq_num+self.seq_num_gap) # Limit of sampling

        # Non-recursive prediction
        forecast_array = np.zeros(shape=(test_total_size,self.seq_num_op,self.state_len))
        true_array = np.zeros(shape=(test_total_size,self.seq_num_op,self.state_len))

        for t in range(test_total_size):
            forecast_array[t] = self.call_inference(test_data[t:t+self.seq_num].reshape(-1,self.seq_num,self.state_len))
            true_array[t] = test_data[t+self.seq_num+self.seq_num_gap:t+self.seq_num+self.seq_num_gap+self.seq_num_op]

        # Rescale
        if self.scaling == 1:
            for lead_time in range(forecast_array.shape[1]):
                forecast_array[:,lead_time,:] = forecast_array[:,lead_time,:] - self.trans_factor
                forecast_array[:,lead_time,:] = forecast_array[:,lead_time,:]*self.trans_factor

                true_array[:,lead_time,:] = true_array[:,lead_time,:] - self.trans_factor
                true_array[:,lead_time,:] = true_array[:,lead_time,:]*self.trans_factor
        else:
            for lead_time in range(forecast_array.shape[1]):
                forecast_array[:,lead_time,:] = self.preproc_pipeline.inverse_transform(forecast_array[:,lead_time,:])
                true_array[:,lead_time,:] = self.preproc_pipeline.inverse_transform(true_array[:,lead_time,:])


        return true_array, forecast_array

    def variational_inference(self,test_data,train_fields,test_fields,pod_modes,training_mean):
        # Restore from checkpoint
        self.restore_model()

        # Scale testing data
        if self.scaling == 1:
            test_data = test_data/self.scale_factor
            test_data = test_data + self.trans_factor
        else:
            test_data = self.preproc_pipeline.transform(test_data)

        # Test data has to be scaled already
        test_total_size = np.shape(test_data)[0]-int(self.seq_num_op+self.seq_num+self.seq_num_gap) # Limit of sampling

        # Non-recursive prediction
        forecast_array = np.zeros(shape=(test_total_size,self.seq_num_op,self.state_len))
        true_array = np.zeros(shape=(test_total_size,self.seq_num_op,self.state_len))

        # Get fixed min/max here
        mean_val, var_val, min_val, max_val = np.mean(train_fields), np.var(train_fields), np.min(train_fields), np.max(train_fields)

        # Remove mean
        test_fields = test_fields - training_mean[None,:]
        
        # Random observation locations
        rand_idx = np.arange(test_fields.shape[1])
        np.random.shuffle(rand_idx)
        rand_idx = rand_idx[:self.num_obs]

        num_grid_points = test_fields.shape[1]
        num_observations = rand_idx.shape[0]
        total_dof = num_grid_points + num_observations

        true_observations = test_fields[:,rand_idx]

        # Non-recursive prediction
        forecast_array = np.zeros(shape=(test_total_size,self.seq_num_op,self.state_len))
        true_array = np.zeros(shape=(test_total_size,self.seq_num_op,self.state_len))
        obj_array = np.zeros(shape=(test_total_size,5))

        # Define residual
        x_ti_rec = None; y_ = None
        def residual(x):
            # Prior
            x = x.reshape(self.seq_num,-1)

            # Scale testing data
            if self.scaling == 1:
                xphys = x/self.scale_factor
                xphys = xphys + self.trans_factor
            else:
                xphys = self.preproc_pipeline.inverse_transform(x)
            
            x_star_rec = np.matmul(pod_modes,xphys.T)#[:,0:1]

            # Likelihood
            x = x.reshape(1,self.seq_num,-1)

            # Scale
            if self.scaling == 1:
                x_tf = self.call_inference(x).numpy()[0].reshape(self.seq_num_op,-1)/self.scale_factor
                x_tf = x_tf + self.trans_factor

            else:
                x_tf = self.preproc_pipeline.inverse_transform(self.call_inference(x).numpy()[0].reshape(self.seq_num_op,-1))

            x_tf_rec = np.matmul(pod_modes,x_tf.T)

            # Sensor predictions
            h_ = x_tf_rec[rand_idx,:].T

            # J
            pred = (np.sum(0.5*(x_star_rec - x_ti_rec)**2)) + (np.sum(0.5*(y_-h_)**2))
            
            return (pred-min_val)/(5000*(max_val-min_val))

        def residual_split(x):
            # Prior
            x = x.reshape(self.seq_num,-1)
            
            # Scale testing data
            if self.scaling == 1:
                xphys = x/self.scale_factor
                xphys = xphys + self.trans_factor
            else:
                xphys = self.preproc_pipeline.inverse_transform(x)

            x_star_rec = np.matmul(pod_modes,xphys.T)#[:,0:1]

            # Likelihood
            x = x.reshape(1,self.seq_num,-1)
            
            if self.scaling == 1:
                x_tf = self.call_inference(x).numpy()[0].reshape(self.seq_num_op,-1)/self.scale_factor
                x_tf = x_tf + self.trans_factor

            else:
                x_tf = self.preproc_pipeline.inverse_transform(self.call_inference(x).numpy()[0].reshape(self.seq_num_op,-1))

            x_tf_rec = np.matmul(pod_modes,x_tf.T)

            # Sensor predictions
            h_ = x_tf_rec[rand_idx,:].T

            # J
            pred = (np.sum(0.5*(x_star_rec - x_ti_rec)**2)) + (np.sum(0.5*(y_-h_)**2))
            
            return pred, np.sum(0.5*(x_star_rec - x_ti_rec)**2), np.sum(0.5*(y_-h_)**2)

        # Define gradient of residual
        def residual_gradient(x):
            # Prior
            x = x.reshape(self.seq_num,-1).astype('double')
            
            # Scale testing data
            if self.scaling == 1:
                xphys = x/self.scale_factor
                xphys = xphys + self.trans_factor
            else:
                xphys = self.preproc_pipeline.inverse_transform(x)

            tf_x_star_rec = tf.convert_to_tensor(np.matmul(pod_modes,xphys.T),dtype='float64')#[:,0:1]
            tf_x_ti_rec = tf.convert_to_tensor(x_ti_rec,dtype='float64')
            tf_y_ = tf.convert_to_tensor(y_,dtype='float64')

            # Likelihood
            x = x.reshape(1,-1)
            x = tf.convert_to_tensor(x,dtype='float64')
            tf_pod_modes = tf.convert_to_tensor(pod_modes,dtype='float64')

            # For both minmax, stdscaler
            # std_scaler = self.preproc_pipeline.get_params()['steps'][0][1]
            # minmax_scaler = self.preproc_pipeline.get_params()['steps'][1][1]

            if self.scaling == 0:
                minmax_scaler = self.preproc_pipeline.get_params()['steps'][0][1]

            with tf.GradientTape(persistent=True) as t:
                t.watch(x)

                x = tf.reshape(x,shape=[1,self.seq_num,-1])
                op = self.call_inference(x)[0]

                # Unscale
                if self.scaling == 1:
                    op = op/self.scale_factor
                    op = op + self.trans_factor
                else:
                    # For both minmax, stdscaler
                    # op = (op+1)/2.0*(minmax_scaler.data_max_- minmax_scaler.data_min_) + minmax_scaler.data_min_
                    # op = (op)*std_scaler.scale_ + std_scaler.mean_
                    op = (op+1)/2.0*(minmax_scaler.data_max_- minmax_scaler.data_min_) + minmax_scaler.data_min_
                
                op = tf.cast(op,dtype='float64')
                x_tf_rec = tf.matmul(tf_pod_modes,tf.transpose(op))


                # Sensor predictions
                tf_idx = tf.convert_to_tensor(rand_idx,dtype='int32')
                h_ = tf.transpose(tf.gather(x_tf_rec,tf_idx))

                # J
                pred = (tf.math.reduce_sum(0.5*(tf_x_star_rec - tf_x_ti_rec)**2)) + \
                        (tf.math.reduce_sum(0.5*(tf_y_-h_)**2))

                pred = (pred-min_val)/(5000*(max_val-min_val))

            grad = t.gradient(pred, x).numpy()[0,:,:].flatten().astype('double')
             
            return grad

        # 3D-Variational update
        var_time = self.var_duration
        for t in range(var_time):

            # Background vector - initial time window input
            x_input = test_data[t:t+self.seq_num].reshape(self.seq_num,self.state_len)

            # Scale testing data
            if self.scaling == 1:
                x_ti = x_input/self.scale_factor
                x_ti = x_ti + self.trans_factor
            else:
                x_ti = self.preproc_pipeline.inverse_transform(x_input)

            x_ti_rec = np.matmul(pod_modes,x_ti.T)#[:,0:1]

            # Observation
            y_ = true_observations[t+self.seq_num+self.seq_num_gap:t+self.seq_num+self.seq_num_gap+self.seq_num_op]

            # Perform optimization
            solution = minimize(residual,x_input.flatten(), jac=residual_gradient, method='SLSQP',
                tol=1e-3,options={'disp': True, 'maxiter': 20, 'eps': 1.4901161193847656e-8})

            old_of, old_bg, old_ll = residual_split(x_input.flatten())
            new_of, new_bg, new_ll = residual_split(solution.x)

            print('Initial guess residual:',old_of, ', Final guess residual:',new_of)
            print('Initial background:',old_bg,'Final background:',new_bg)
            print('Initial Likelihood:',old_ll,'Final Likelihood:',new_ll)

            obj_array[t,0] = old_bg
            obj_array[t,1] = old_ll
            obj_array[t,2] = new_bg
            obj_array[t,3] = new_ll

            if new_of< old_of:
                assimilated_rec_input_seq = solution.x.reshape(1,self.seq_num,-1)
                forecast_array[t] = self.call_inference(assimilated_rec_input_seq).numpy()[0]
                obj_array[t,4] = 1
            else:
                print('Optimization failed. Initial guess residual:',old_of, ', Final guess residual:',new_of)
                x_input = x_input.reshape(1,self.seq_num,-1)
                forecast_array[t] = self.call_inference(x_input).numpy()[0]

            # Recording truth            
            true_array[t] = test_data[t+self.seq_num+self.seq_num_gap:t+self.seq_num+self.seq_num_gap+self.seq_num_op]
            
            print('Finished variational prediction for timestep: ',t)

        # Rescale
        if self.scaling == 1:
            for lead_time in range(forecast_array.shape[1]):
                forecast_array[:,lead_time,:] = forecast_array[:,lead_time,:] - self.trans_factor
                forecast_array[:,lead_time,:] = forecast_array[:,lead_time,:]*self.trans_factor

                true_array[:,lead_time,:] = true_array[:,lead_time,:] - self.trans_factor
                true_array[:,lead_time,:] = true_array[:,lead_time,:]*self.trans_factor
        else:
            for lead_time in range(forecast_array.shape[1]):
                forecast_array[:,lead_time,:] = self.preproc_pipeline.inverse_transform(forecast_array[:,lead_time,:])
                true_array[:,lead_time,:] = self.preproc_pipeline.inverse_transform(true_array[:,lead_time,:])

        return true_array, forecast_array, obj_array

    def constrained_variational_inference(self,test_data,train_fields,test_fields,pod_modes,training_mean,num_fixed_modes):
        # Restore from checkpoint
        self.restore_model()

        # Scale testing data
        if self.scaling == 1:
            test_data = test_data/self.scale_factor
            test_data = test_data + self.trans_factor
        else:
            test_data = self.preproc_pipeline.transform(test_data)

        # Test data has to be scaled already
        test_total_size = np.shape(test_data)[0]-int(self.seq_num+self.seq_num_op+self.seq_num_gap) # Limit of sampling

        # Non-recursive prediction
        forecast_array = np.zeros(shape=(test_total_size,self.seq_num_op,self.state_len))
        true_array = np.zeros(shape=(test_total_size,self.seq_num_op,self.state_len))

        # Get fixed min/max here
        mean_val, var_val, min_val, max_val = np.mean(train_fields), np.var(train_fields), np.min(train_fields), np.max(train_fields)

        # Remove mean
        test_fields = test_fields - training_mean[None,:]
        
        # Random observation locations
        rand_idx = np.arange(test_fields.shape[1])
        np.random.shuffle(rand_idx)
        rand_idx = rand_idx[:self.num_obs]

        num_grid_points = test_fields.shape[0]
        num_observations = rand_idx.shape[0]
        total_dof = num_grid_points + num_observations

        true_observations = test_fields[:,rand_idx]

        # Non-recursive prediction
        forecast_array = np.zeros(shape=(test_total_size,self.seq_num_op,self.state_len))
        true_array = np.zeros(shape=(test_total_size,self.seq_num_op,self.state_len))

        # # Fixed modes
        # all_modes = np.arange(pod_modes.shape[1])
        # variable_modes = np.asarray(variable_modes)
        # fixed_modes = numpy.setxor1d(all_modes, variable_modes)

        # Define residual
        x_ti_rec = None; y_ = None; 
        global x_fixed
        x_fixed = None
        # x_fixed = test_data[:self.seq_num,:num_fixed_modes]
        def residual(x):
            global x_fixed
            # Prior
            x_fixed = x_fixed.reshape(self.seq_num,-1)
            x = x.reshape(self.seq_num,-1)
            x = np.concatenate((x_fixed,x),axis=1)

            # Scale testing data
            if self.scaling == 1:
                xphys = x/self.scale_factor
                xphys = xphys + self.trans_factor
            else:
                xphys = self.preproc_pipeline.inverse_transform(x)

            x_star_rec = np.matmul(pod_modes,xphys.T)#[:,0:1]

            # Likelihood
            x = x.reshape(1,self.seq_num,-1)
            
            # Scale
            if self.scaling == 1:
                x_tf = self.call_inference(x).numpy()[0].reshape(self.seq_num_op,-1)/self.scale_factor
                x_tf = x_tf + self.trans_factor
            else:
                x_tf = self.preproc_pipeline.inverse_transform(self.call_inference(x).numpy()[0].reshape(self.seq_num_op,-1))

            x_tf_rec = np.matmul(pod_modes,x_tf.T)

            # Sensor predictions
            h_ = x_tf_rec[rand_idx,:].T

            # J
            pred = (np.sum(0.5*(x_star_rec - x_ti_rec)**2)) + (np.sum(0.5*(y_-h_)**2))
            
            return (pred-min_val)/(5000*(max_val-min_val))

        # Define gradient of residual
        def residual_gradient(x_var):
            global x_fixed
            x_var = x_var.reshape(self.seq_num,-1)
            x = np.concatenate((x_fixed,x_var),axis=1)

            # Scale testing data
            if self.scaling == 1:
                xphys = x/self.scale_factor
                xphys = xphys + self.trans_factor
            else:
                xphys = self.preproc_pipeline.inverse_transform(x)

            tf_x_star_rec = tf.convert_to_tensor(np.matmul(pod_modes,xphys.T),dtype='float64')#[:,0:1]
            tf_x_ti_rec = tf.convert_to_tensor(x_ti_rec,dtype='float64')
            tf_y_ = tf.convert_to_tensor(y_,dtype='float64')

            # Likelihood
            # x = x.reshape(1,-1)
            # x = tf.convert_to_tensor(x,dtype='float64')

            x_var_tf = x_var.reshape(1,-1)
            x_var_tf = tf.convert_to_tensor(x_var_tf,dtype='float64')

            x_fixed_tf = x_fixed.reshape(1,-1)
            x_fixed_tf = tf.convert_to_tensor(x_fixed_tf,dtype='float64')
            tf_pod_modes = tf.convert_to_tensor(pod_modes,dtype='float64')

            # For both minmax, stdscaler
            # std_scaler = self.preproc_pipeline.get_params()['steps'][0][1]
            # minmax_scaler = self.preproc_pipeline.get_params()['steps'][1][1]

            if self.scaling == 0:
                minmax_scaler = self.preproc_pipeline.get_params()['steps'][0][1]

            with tf.GradientTape(persistent=True) as t:
                t.watch(x_var_tf)

                x = tf.concat([x_fixed_tf,x_var_tf],axis=1)
                x = tf.reshape(x,shape=[1,self.seq_num,-1])
                op = self.call_inference(x)[0]

                # Unscale
                if self.scaling == 1:
                    op = op/self.scale_factor
                    op = op + self.trans_factor
                else:
                    # For both minmax, stdscaler
                    # op = (op+1)/2.0*(minmax_scaler.data_max_- minmax_scaler.data_min_) + minmax_scaler.data_min_
                    # op = (op)*std_scaler.scale_ + std_scaler.mean_
                    op = (op+1)/2.0*(minmax_scaler.data_max_- minmax_scaler.data_min_) + minmax_scaler.data_min_
                
                op = tf.cast(op,dtype='float64')
                x_tf_rec = tf.matmul(tf_pod_modes,tf.transpose(op))


                # Sensor predictions
                tf_idx = tf.convert_to_tensor(rand_idx,dtype='int32')
                h_ = tf.transpose(tf.gather(x_tf_rec,tf_idx))

                # J
                pred = (tf.math.reduce_sum(0.5*(tf_x_star_rec - tf_x_ti_rec)**2)) + \
                        (tf.math.reduce_sum(0.5*(tf_y_-h_)**2))

                pred = (pred-min_val)/(5000*(max_val-min_val))

            grad = t.gradient(pred, x_var_tf).numpy().flatten().astype('double')
             
            return grad

        # 3D-Variational update
        var_time = self.var_duration
        for t in range(var_time):

            # Background vector - initial time window input
            x_input = test_data[t:t+self.seq_num].reshape(self.seq_num,self.state_len)
            
            # Fix some scales
            x_fixed = x_input[:,:num_fixed_modes]
            x_var = x_input[:,num_fixed_modes:]

            # Scale testing data
            if self.scaling == 1:
                x_ti = x_input/self.scale_factor
                x_ti = x_ti + self.trans_factor
            else:
                x_ti = self.preproc_pipeline.inverse_transform(x_input)

            x_ti_rec = np.matmul(pod_modes,x_ti.T)#[:,0:1]

            # Observation
            y_ = true_observations[t+self.seq_num+self.seq_num_gap:t+self.seq_num+self.seq_num_gap+self.seq_num_op]

            # Perform optimization
            solution = minimize(residual,x_var.flatten(), jac=residual_gradient, method='SLSQP',
                tol=1e-3,options={'disp': True, 'maxiter': 20, 'eps': 1.4901161193847656e-8})

            old_of = residual(x_var.flatten())
            new_of = residual(solution.x)

            print('Initial guess residual:',old_of, ', Final guess residual:',new_of)

            if new_of< old_of:
                xtemp = solution.x.reshape(self.seq_num,-1)
                x_solution = np.concatenate((x_fixed,xtemp),axis=1).reshape(1,self.seq_num,-1)
                forecast_array[t] = self.call_inference(x_solution).numpy()[0]

            else:

                print('Optimization failed. Initial guess residual:',old_of, ', Final guess residual:',new_of)

                x_input = x_input.reshape(1,self.seq_num,-1)
                forecast_array[t] = self.call_inference(x_input).numpy()[0]
            
            true_array[t] = test_data[t+self.seq_num+self.seq_num_gap:t+self.seq_num+self.seq_num_gap+self.seq_num_op]
            
            print('Finished variational prediction for timestep: ',t)

        # Rescale
        if self.scaling == 1:
            for lead_time in range(forecast_array.shape[1]):
                forecast_array[:,lead_time,:] = forecast_array[:,lead_time,:] - self.trans_factor
                forecast_array[:,lead_time,:] = forecast_array[:,lead_time,:]*self.trans_factor

                true_array[:,lead_time,:] = true_array[:,lead_time,:] - self.trans_factor
                true_array[:,lead_time,:] = true_array[:,lead_time,:]*self.trans_factor
        else:
            for lead_time in range(forecast_array.shape[1]):
                forecast_array[:,lead_time,:] = self.preproc_pipeline.inverse_transform(forecast_array[:,lead_time,:])
                true_array[:,lead_time,:] = self.preproc_pipeline.inverse_transform(true_array[:,lead_time,:])



        return true_array, forecast_array


if __name__ == '__main__':
    print('Architecture file')
