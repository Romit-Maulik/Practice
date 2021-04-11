import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions

# Set seeds
np.random.seed(10)
tf.random.set_seed(10)

from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Dropout
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import Model
from tensorflow.keras import backend as K

from sklearn.preprocessing import StandardScaler, MinMaxScaler
tf.keras.backend.set_floatx('float32')

class weight_averaging(Model):
    def __init__(self,data_tuple,lrate=0.001,num_epochs=1000):
        super(weight_averaging, self).__init__()

        train_data = data_tuple[0]
        valid_data = data_tuple[1]
        test_data = data_tuple[2]
        
        self.input_train_data = train_data[:,0].reshape(-1,1)
        self.input_valid_data = valid_data[:,0].reshape(-1,1)
        self.input_test_data = test_data[:,0].reshape(-1,1)
        
        self.output_train_data = train_data[:,1].reshape(-1,1)
        self.output_valid_data = valid_data[:,1].reshape(-1,1)
        self.output_test_data = train_data[:,1].reshape(-1,1)

        self.ntrain = self.input_train_data.shape[0]
        self.nvalid = self.input_valid_data.shape[0]
        self.ntest = self.input_test_data.shape[0]
        self.num_latent = 6

        self.init_architecture()
        self.train_op = tf.keras.optimizers.Adam(learning_rate=lrate)
        
        # num epochs
        self.num_epochs = num_epochs

    def init_architecture(self):

        # Define model architecture
        self.l1 = Dense(50,activation='relu')
        self.out_mean = Dense(1,activation='linear')
        self.out_logvar = Dense(1,activation='linear') #sigma^2
    

    def call(self,X):

        hh = self.l1(X)
        mean = self.out_mean(hh)
        logvar = self.out_logvar(hh)

        return mean, logvar
    
    # negative log likelihood
    def get_loss(self,X,Y):
        op_mean, op_logvar = self.call(X)
            
        op_var = tf.math.exp(op_logvar)
        half_logvar = 0.5*op_logvar
        
        mse = (tf.math.square(op_mean-Y))*0.5/(op_var+K.epsilon())
        loss = tf.reduce_mean(half_logvar+mse)
        
        return loss
    
    # perform gradient descent - regular
    def network_learn(self,X,Y):
        g = self.get_grad(X,Y)
        self.train_op.apply_gradients(zip(g, self.trainable_variables))

    # get gradients - regular
    def get_grad(self,X,Y):
        # Regular training
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(X,Y)
            g = tape.gradient(L, self.trainable_variables)            
        
        return g

    # Train the model
    def train_model(self):
        plot_iter = 0
        stop_iter = 0
        patience = 100
        best_valid_loss = np.inf # Some large number 
        swa_iter = 0

        self.num_batches = 1
        self.train_batch_size = int(self.ntrain/self.num_batches)
        self.valid_batch_size = int(self.nvalid/self.num_batches)
        
        for i in range(self.num_epochs):
            # Training loss
            print('Training iteration:',i)
            
            for batch in range(self.num_batches):
                input_batch = self.input_train_data[batch*self.train_batch_size:(batch+1)*self.train_batch_size]
                output_batch = self.output_train_data[batch*self.train_batch_size:(batch+1)*self.train_batch_size]
                self.network_learn(input_batch,output_batch)

            # Validation loss
            valid_loss = 0.0

            for batch in range(self.num_batches):
                input_batch = self.input_valid_data[batch*self.valid_batch_size:(batch+1)*self.valid_batch_size]
                output_batch = self.output_valid_data[batch*self.valid_batch_size:(batch+1)*self.valid_batch_size]
                valid_loss = valid_loss + self.get_loss(input_batch,output_batch).numpy()

            valid_loss = valid_loss/self.nvalid

            # Check early stopping criteria
            if valid_loss < best_valid_loss:
                
                print('Improved validation loss from:',best_valid_loss,' to:', valid_loss)                
                best_valid_loss = valid_loss
                self.save_weights('./checkpoints/weight_averaging_checkpoint')
                stop_iter = 0
            else:
                print('Validation loss (no improvement):',valid_loss)
                stop_iter = stop_iter + 1

            if stop_iter == patience:
                if swa_iter < 5:
                    self.save_weights('./checkpoints/weight_averaging_checkpoint_'+str(swa_iter))
                    swa_iter +=1
                
                    stop_iter = 0    
                else:
                    break
                
    # Do some testing
    def model_inference(self):
        # Average weights - initialization
        self.load_weights('./checkpoints/weight_averaging_checkpoint')
        
        mean, logvar = self.call(self.input_test_data)
        mean = mean.numpy()
        var = np.exp(logvar.numpy()) + mean**2
        
        for i in range(5):
            self.load_weights('./checkpoints/weight_averaging_checkpoint_'+str(i))
        
            t1, t2 = self.call(self.input_test_data)
            mean = mean + t1.numpy()
            var = var + np.exp(t2.numpy()) + mean**2
            
        ensemble_mean = mean/6.0
        ensemble_var = var/6.0 - ensemble_mean**2
        ensemble_std = np.sqrt(ensemble_var)
        
        np.save('Weight_averaging_mean.npy',ensemble_mean)
        np.save('Weight_averaging_std.npy',ensemble_std)
        
        return None
        


if __name__ == '__main__':
    print('Weight averaging model architecture')