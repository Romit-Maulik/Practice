import os
dir_path = os.path.dirname(os.path.realpath(__file__))

import numpy as np
np.random.seed(10)
import tensorflow as tf
tf.random.set_seed(10)
from tensorflow.keras import Model
from flow_layers import scalar_real_nvp

# Plotting
import matplotlib.pyplot as plt

#Build the model which does basic map of inputs to coefficients
class normalizing_flow(Model):
    def __init__(self,data,params):
        super(normalizing_flow, self).__init__()

        self.dim = data.shape[1]
        self.data = data
        self.params = params

        # Define real_nvp flow layers
        self.l0 = scalar_real_nvp()
        self.l1 = scalar_real_nvp()
        self.l2 = scalar_real_nvp()
        self.l3 = scalar_real_nvp()

        # Training optimizer
        self.train_op = tf.keras.optimizers.Adam(learning_rate=0.0001)

    @tf.function
    def call(self, x, params):

        h1, ldj1 = self.l0(x,params) # Individual layer calls
        h2, ldj2 = self.l1(h1,params)
        h3, ldj3 = self.l2(h2,params)
        h4, ldj4 = self.l3(h3,params)

        logdet = ldj1+ldj2+ldj3+ldj4

        log_prior = -0.5*tf.math.reduce_sum(tf.math.square(h4))

        neg_ll = - log_prior - logdet

        return h4, neg_ll

    @tf.function
    def sample(self,size,params):
        z = tf.random.normal(shape=(size,self.dim))

        h = self.l3.invert(z,params) # Individual layer calls
        h = self.l2.invert(h,params) # Individual layer calls
        h = self.l1.invert(h,params) # Individual layer calls
        h = self.l0.invert(h,params) # Individual layer calls

        return h

    # perform gradient descent
    def network_learn(self,x,params):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            _, neg_ll = self.call(x,params)
            g = tape.gradient(neg_ll, self.trainable_variables)

        self.train_op.apply_gradients(zip(g, self.trainable_variables))

    # Train the model
    def train_model(self):
        plot_iter = 0
        stop_iter = 0
        patience = 10
        best_valid_loss = np.inf # Some large number 

        self.num_batches = 10
        self.ntrain = int(0.7*self.data.shape[0])
        self.nvalid = self.data.shape[0] - int(0.7*self.data.shape[0])

        self.train_data = self.data[:self.ntrain]
        self.train_params = self.params[:self.ntrain]

        self.valid_data = self.data[self.ntrain:]
        self.valid_params = self.params[self.ntrain:]

        self.train_batch_size = int(self.ntrain/self.num_batches)
        self.valid_batch_size = int(self.ntrain/self.num_batches)
        
        for i in range(2000):
            # Training loss
            print('Training iteration:',i)
            
            for batch in range(self.num_batches):
                batch_data = self.train_data[batch*self.train_batch_size:(batch+1)*self.train_batch_size]
                batch_params = self.train_params[batch*self.train_batch_size:(batch+1)*self.train_batch_size]
                self.network_learn(batch_data,batch_params)

            # Validation loss
            valid_loss = 0.0

            for batch in range(self.num_batches):
                batch_data = self.valid_data[batch*self.valid_batch_size:(batch+1)*self.valid_batch_size]
                batch_params = self.valid_params[batch*self.train_batch_size:(batch+1)*self.train_batch_size]
                valid_loss = valid_loss + np.sum(self.call(batch_data,batch_params)[1].numpy())

            # Check early stopping criteria
            if valid_loss < best_valid_loss:
                
                print('Improved validation negative log likelihood from:',best_valid_loss,' to:', valid_loss)
                
                best_valid_loss = valid_loss

                self.save_weights('./checkpoints/my_checkpoint')
                
                stop_iter = 0
            else:
                print('Validation negative log likelihood (no improvement):',valid_loss)
                stop_iter = stop_iter + 1

            if stop_iter == patience:
                break


if __name__ == '__main__':
    train_mode = False

    # Load data from target
    data = np.load('Training_source.npy')
    # Load data for parameters
    params = np.load('Training_stencil.npy')
    
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)

    train_data = data[idx[:500000]]
    train_params = params[idx[:500000]]

    test_data = data[idx[500000:]]
    test_params = params[idx[500000:]]

    # Scaling makes it easier to train
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Normalizing flow training
    flow_model = normalizing_flow(train_data,train_params)
    pre_samples = flow_model.sample(test_data.shape[0],test_params).numpy()
   
    if train_mode:
        flow_model.train_model()
    else:
        flow_model.load_weights('./checkpoints/my_checkpoint')

    samples = flow_model.sample(test_data.shape[0],test_params).numpy()

    plt.figure()
    plt.hist(test_data[:,0],label='Target',density=True)
    plt.hist(pre_samples[:,0],label='Before training', density=True)
    plt.legend()


    plt.figure()
    plt.hist(data[:,0],label='Target',density=True)
    plt.hist(samples[:,0],label='Generated',density=True)
    plt.legend()
    plt.show()
    
