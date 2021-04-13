import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

import tensorflow as tf
tf.random.set_seed(10)

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

# Lyapunov
from nolds import lyap_r


#Build the model which does basic map of inputs to coefficients
class standard_lstm(Model):
    def __init__(self,data,lce=False,multistep=False):
        super(standard_lstm, self).__init__()

        # Is this an LCE based LSTM
        self.lce = lce
        self.multistep = multistep
        self.num_multistep = 5

        # Set up the data for the LSTM
        self.data_tsteps = np.shape(data)[0]
        self.state_len = np.shape(data)[1]

        self.preproc_pipeline = Pipeline([('stdscaler', StandardScaler()),('minmax', MinMaxScaler(feature_range=(-1, 1)))])
        self.data = self.preproc_pipeline.fit_transform(data)

        # Need to make minibatches
        self.seq_num = 1

        if not self.multistep:
            self.total_size = np.shape(data)[0]-int(self.seq_num) # Limit of sampling

            input_seq = np.zeros(shape=(self.total_size,self.seq_num,self.state_len))  #[samples,n_inputs,state_len]
            output_seq = np.zeros(shape=(self.total_size,self.state_len)) #[samples,n_outputs,state_len]

            snum = 0
            for t in range(0,self.total_size):
                input_seq[snum,:,:] = self.data[None,t:t+self.seq_num,:]
                output_seq[snum,:] = self.data[None,t+self.seq_num,:]        
                snum = snum + 1

        else:
            self.total_size = np.shape(data)[0]- self.seq_num - self.num_multistep # Limit of sampling

            input_seq = np.zeros(shape=(self.total_size,self.seq_num,self.state_len))  #[samples,n_inputs,state_len]
            output_seq = np.zeros(shape=(self.total_size,self.num_multistep, self.state_len)) #[samples,n_outputs,state_len]

            snum = 0
            for t in range(0,self.total_size):
                input_seq[snum,:,:] = self.data[None,t:t+self.seq_num,:]
                for step in range(self.num_multistep):
                    output_seq[snum,step,:] = self.data[None,t+self.seq_num+step,:]        
                
                snum = snum + 1


        # Shuffle dataset
        idx = np.arange(snum)
        np.random.shuffle(idx)
        input_seq = input_seq[idx]
        output_seq = output_seq[idx]

        # Split into train and test
        self.input_seq_test = input_seq[int(0.9*snum):]
        self.output_seq_test = output_seq[int(0.9*snum):]
        input_seq = input_seq[:int(0.9*snum)]
        output_seq = output_seq[:int(0.9*snum)]

        # Split into train and valid
        self.ntrain = int(0.8*np.shape(input_seq)[0])
        self.nvalid = np.shape(input_seq)[0] - self.ntrain

        self.input_seq_train = input_seq[:self.ntrain]
        self.output_seq_train = output_seq[:self.ntrain]

        self.input_seq_valid = input_seq[self.ntrain:]
        self.output_seq_valid = output_seq[self.ntrain:]

        # Define architecture
        xavier=tf.keras.initializers.GlorotUniform()
        self.l1=tf.keras.layers.LSTM(50,return_sequences=True,input_shape=(self.seq_num,self.state_len))
        self.l2=tf.keras.layers.LSTM(50,return_sequences=False)
        self.out = tf.keras.layers.Dense(self.state_len)
        self.train_op = tf.keras.optimizers.Adam(learning_rate=0.001)

        if self.lce:
            # Find Lyapunov exponents of the true data
            self.ly_exps = np.zeros(shape=(self.state_len),dtype='double')
            for i in range(self.state_len):
                self.ly_exps[i] = lyap_r(self.data[:,i])
    
    # Running the model
    def call(self,X):
        h1 = self.l1(X)
        h2 = self.l2(h1)
        out = self.out(h2)
        return out
    
    # Regular MSE
    def get_loss(self,X,Y):

        if not self.multistep:
            
            op=self.call(X)
            return tf.reduce_mean(tf.math.square(op-Y))
        
        else:
            for i in range(self.num_multistep):
                op = self.call(X)
                op = tf.expand_dims(op,axis=1)
                X = tf.concat([X,op],axis=1)
                X = X[:,1:]

                if i == 0:
                    Y_pred = op
                else:
                    Y_pred = tf.concat([Y_pred,op],axis=1)

            return tf.reduce_mean(tf.math.square(Y_pred-Y))


    # get gradients - regular
    def get_grad(self,X,Y):
        if self.lce:

            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                L = self.get_loss(X,Y)

                ran = np.random.uniform(low=0.0,high=1.0)

                if ran > 0.5:
                    # Make a recursive call and calculate LCE
                    # Input placeholder setup
                    rec_input_seq = self.data[:self.seq_num,:].reshape(1,self.seq_num,self.state_len)
                    
                    # True outputs
                    rec_output_seq = np.zeros(shape=(60,self.state_len)) #[samples,n_outputs,state_len]
                    t_init = np.random.randint(low=0,high=self.total_size-60)

                    snum = 0
                    for t in range(t_init,t_init+60):# Starting after limit cycle
                        rec_output_seq[snum,:] = self.data[None,t+self.seq_num,:]        
                        snum = snum + 1

                    # Recursive predict
                    rec_pred = np.copy(rec_output_seq)
                    for t in range(60):
                        rec_pred[t] = self.call(rec_input_seq).numpy()[0]
                        rec_input_seq[0,0:-1,:] = rec_input_seq[0,1:]
                        rec_input_seq[0,-1,:] = rec_pred[t]

                    # rec_loss = tf.reduce_sum(tf.square(tf.convert_to_tensor(rec_output_seq-rec_pred,dtype=tf.float32)))

                    # Find Lyapunov exponents of the predicted data
                    pred_ly_exps = np.zeros(shape=(self.state_len),dtype='double')
                    for i in range(self.state_len):
                        pred_ly_exps[i] = lyap_r(rec_pred[:,i])

                    ly_loss = 100.0*tf.reduce_sum(tf.square(tf.convert_to_tensor(pred_ly_exps-self.ly_exps,dtype=tf.float32)))
                    
                    # Calculate gradient
                    g = tape.gradient(L+ly_loss, self.trainable_variables) # Enhanced trainable variables

                else:

                    g = tape.gradient(L, self.trainable_variables)

            return g

        else:
            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                L = self.get_loss(X,Y)
                g = tape.gradient(L, self.trainable_variables)
            return g
    
    # perform gradient descent - regular
    def network_learn(self,X,Y):
        g = self.get_grad(X,Y)
        self.train_op.apply_gradients(zip(g, self.trainable_variables))

    # Train the model
    def train_model(self):
        plot_iter = 0
        stop_iter = 0
        patience = 10
        best_valid_loss = 999999.0 # Some large number 

        self.num_batches = 20
        self.train_batch_size = int(self.ntrain/self.num_batches)
        self.valid_batch_size = int((self.nvalid)/self.num_batches)
        
        for i in range(100):
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

                valid_loss = valid_loss + self.get_loss(input_batch,output_batch).numpy()

                if self.multistep:
                    X = self.input_seq_valid
                    
                    for i in range(self.num_multistep):
                        predictions = self.call(X)
                        predictions = tf.expand_dims(predictions,axis=1)
                        X = tf.concat([X,predictions],axis=1)
                        X = X[:,1:]

                    valid_r2 = valid_r2 + coeff_determination(predictions,self.output_seq_valid[:,-1,:])
                else:
                    predictions = self.call(self.input_seq_valid)
                    valid_r2 = valid_r2 + coeff_determination(predictions,self.output_seq_valid)

            valid_r2 = valid_r2/(batch+1)


            # Check early stopping criteria
            if valid_loss < best_valid_loss:
                
                print('Improved validation loss from:',best_valid_loss,' to:', valid_loss)
                print('Validation R2:',valid_r2)
                
                best_valid_loss = valid_loss

                if self.lce:
                    self.save_weights('./lce_checkpoints/my_checkpoint')
                elif self.multistep:
                    self.save_weights('./multistep_checkpoints/my_checkpoint')
                else:
                    self.save_weights('./checkpoints/my_checkpoint')
                
                stop_iter = 0
            else:
                print('Validation loss (no improvement):',valid_loss)
                print('Validation R2:',valid_r2)
                stop_iter = stop_iter + 1

            if stop_iter == patience:
                break


    # Load weights
    def restore_model(self):
        if self.lce:
            self.load_weights(dir_path+'/lce_checkpoints/my_checkpoint') # Load pretrained model
        elif self.multistep:
            self.load_weights(dir_path+'/multistep_checkpoints/my_checkpoint') # Load pretrained model
        else:
            self.load_weights(dir_path+'/checkpoints/my_checkpoint') # Load pretrained model

    # Do some testing
    def model_inference(self,test_data):
        # Restore from checkpoint
        self.restore_model()

        # Scale testing data
        test_data = self.preproc_pipeline.fit_transform(test_data)

        # Test sizes
        test_total_size = np.shape(test_data)[0]-int(self.seq_num) # Limit of sampling

        # Input placeholder setup
        rec_input_seq = test_data[:self.seq_num,:].reshape(1,self.seq_num,self.state_len)
        
        # True outputs
        rec_output_seq = np.zeros(shape=(test_total_size,self.state_len)) #[samples,n_outputs,state_len]
        snum = 0
        for t in range(test_total_size):
            rec_output_seq[snum,:] = test_data[None,t+self.seq_num,:]        
            snum = snum + 1

        # Recursive predict
        rec_pred = np.copy(rec_output_seq)
        for t in range(test_total_size):
            rec_pred[t] = self.call(rec_input_seq).numpy()[0]
            rec_input_seq[0,0:-1,:] = rec_input_seq[0,1:]
            rec_input_seq[0,-1,:] = rec_pred[t]

        # Rescale
        rec_pred = self.preproc_pipeline.inverse_transform(rec_pred)
        rec_output_seq = self.preproc_pipeline.inverse_transform(rec_output_seq)

        # plot
        for i in range(self.state_len):
            plt.figure()
            plt.title('Mode '+str(i))
            plt.plot(rec_pred[:,i],label='Predicted')
            plt.plot(rec_output_seq[:,i],label='True')
            plt.legend()
            plt.show()

        return rec_output_seq, rec_pred


if __name__ == '__main__':
    print('Architecture file')