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

from utils import posterior_mean_field, prior_trainable, negloglik

class flipout_model(Model):
    def __init__(self,data_tuple,lrate=0.001,num_epochs=1000):
        super(flipout_model, self).__init__()

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

        # for BNN
        kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                            tf.cast(self.ntrain, dtype=tf.float32))

        self.l1 = tfp.layers.DenseFlipout(50, activation='relu',kernel_divergence_fn=kl_divergence_function)
        self.out_mean = tfp.layers.DenseFlipout(1,activation='linear',kernel_divergence_fn=kl_divergence_function)
        self.out_logvar = tfp.layers.DenseFlipout(1,activation='linear',kernel_divergence_fn=kl_divergence_function) #sigma^2
    

    def call(self,X):

        hh = self.l1(X)
        mean = self.out_mean(hh)
        logvar = self.out_logvar(hh)

        return mean, logvar
    
    # Regular MSE
    def get_loss(self,X,Y):
        op_mean, op_logvar = self.call(X)
        op_var = tf.math.exp(op_logvar)
        half_logvar = 0.5*op_logvar

        mse = (tf.math.square(op_mean-Y))*0.5/(op_var+K.epsilon())
        nll = tf.reduce_mean(half_logvar+mse)
        
        loss_kl = tf.reduce_sum(self.losses)/self.ntrain
        
        return nll + loss_kl
    
    # get gradients - regular
    def get_grad(self,X,Y):
        # Regular training
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
                self.save_weights('./checkpoints/flipout_checkpoint')
                stop_iter = 0
            else:
                print('Validation loss (no improvement):',valid_loss)
                stop_iter = stop_iter + 1

            if stop_iter == patience:
                break
                
    # Load weights
    def restore_model(self):
        self.load_weights('./checkpoints/flipout_checkpoint')

    # Do some testing
    def model_inference(self,mc_num=1000):
        # Restore from checkpoint
        self.restore_model()
        
        prediction_list = []
        for i in range(mc_num):
            mean, var = self.call(self.input_test_data)
            mean = mean.numpy()
                          
            prediction_list.append(mean)

        prediction_list = np.asarray(prediction_list)
        
        np.save('Flipout_predictions.npy',prediction_list)


if __name__ == '__main__':
    print('flipout model architecture')