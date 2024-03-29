import os
dir_path = os.path.dirname(os.path.realpath(__file__))

import numpy as np
np.random.seed(10)
import tensorflow as tf
tf.random.set_seed(10)
from tensorflow.keras import Model

# Plotting
import matplotlib.pyplot as plt

from flow_layers import real_nvp

#Build the model which does basic map of inputs to coefficients
class normalizing_flow(Model):
    def __init__(self,data):
        super(normalizing_flow, self).__init__()

        self.dim = data.shape[1]
        self.scale_dim = int(self.dim/2)
        self.data = data

        # Define real_nvp flow layers
        self.l0 = real_nvp(self.scale_dim,shuffle=True)
        self.l1 = real_nvp(self.scale_dim,shuffle=True)
        self.l2 = real_nvp(self.scale_dim,shuffle=True)
        self.l3 = real_nvp(self.scale_dim,shuffle=True)
        self.l4 = real_nvp(self.scale_dim,shuffle=True)
        self.l5 = real_nvp(self.scale_dim,shuffle=True)

        # Training optimizer
        self.train_op = tf.keras.optimizers.Adam(learning_rate=0.0001)


    @tf.function
    def call(self, x):

        h1, ldj1 = self.l0(x) # Individual layer calls
        h2, ldj2 = self.l1(h1)
        h3, ldj3 = self.l2(h2)
        h4, ldj4 = self.l3(h3)
        h5, ldj5 = self.l3(h4)
        h6, ldj6 = self.l3(h5)

        logdet = ldj1+ldj2+ldj3+ldj4+ldj5+ldj6

        log_prior = -0.5*tf.math.reduce_sum(tf.math.square(h6))

        neg_ll = - log_prior - logdet

        return h4, neg_ll

    @tf.function
    def sample(self,size):
        z = tf.random.normal(shape=(size,self.dim))

        h = self.l3.invert(z) # Individual layer calls
        h = self.l2.invert(h) # Individual layer calls
        h = self.l1.invert(h) # Individual layer calls
        h = self.l0.invert(h) # Individual layer calls

        return h

    # perform gradient descent
    def network_learn(self,x):

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            _, neg_ll = self.call(x)
            g = tape.gradient(neg_ll, self.trainable_variables)

        self.train_op.apply_gradients(zip(g, self.trainable_variables))

    # Train the model
    def train_model(self):
        plot_iter = 0
        stop_iter = 0
        patience = 100
        best_valid_loss = np.inf # Some large number 

        self.num_batches = 10
        self.ntrain = int(0.7*self.data.shape[0])
        self.nvalid = self.data.shape[0] - int(0.7*self.data.shape[0])

        self.train_data = self.data[:self.ntrain]
        self.valid_data = self.data[self.ntrain:]

        self.train_batch_size = int(self.ntrain/self.num_batches)
        self.valid_batch_size = int(self.ntrain/self.num_batches)
        
        for i in range(1000):
            # Training loss
            print('Training iteration:',i)
            
            for batch in range(self.num_batches):
                batch_data = self.train_data[batch*self.train_batch_size:(batch+1)*self.train_batch_size]
                self.network_learn(batch_data)

            # Validation loss
            valid_loss = 0.0

            for batch in range(self.num_batches):
                batch_data = self.valid_data[batch*self.valid_batch_size:(batch+1)*self.valid_batch_size]
                valid_loss = valid_loss + np.sum(self.call(batch_data)[1].numpy())

            # Check early stopping criteria
            if valid_loss < best_valid_loss:
                
                print('Improved validation loss from:',best_valid_loss,' to:', valid_loss)
                
                best_valid_loss = valid_loss

                self.save_weights('./checkpoints/my_checkpoint')
                
                stop_iter = 0
            else:
                print('Validation loss (no improvement):',valid_loss)
                stop_iter = stop_iter + 1

            if stop_iter == patience:
                break


if __name__ == '__main__':
    train_mode = True

    # Generate some data from a weird distribution
    from sklearn.datasets import make_moons
    data = make_moons(n_samples=1000,noise=0.1)[0]
  
    # Normalizing flow training
    flow_model = normalizing_flow(data)
    flow_model.build(input_shape=(1,2))
    pre_samples = flow_model.sample(1000).numpy()
    
    if train_mode:
        flow_model.train_model()
    else:
        flow_model.load_weights('./checkpoints/my_checkpoint')

    samples = flow_model.sample(4000).numpy()

    # plt.figure()
    # plt.scatter(data[:,0],data[:,1],label='Target')
    # plt.scatter(pre_samples[:,0],pre_samples[:,1],label='Before training')
    # plt.legend()

    plt.figure()
    plt.scatter(data[:,0],data[:,1],label='Target')
    plt.scatter(samples[:,0],samples[:,1],label='Generated',alpha=0.5)
    plt.legend()
    plt.show()
    
