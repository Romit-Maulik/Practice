import os
dir_path = os.path.dirname(os.path.realpath(__file__))

import numpy as np
np.random.seed(10)
import tensorflow as tf
tf.random.set_seed(10)
from tensorflow.keras import layers

""" 
Build a parametric Real NVP flow layer
"""
class parametric_real_nvp(layers.Layer):
    def __init__(self, scale_dim, num_dims, shuffle=False):
        super(parametric_real_nvp, self).__init__()

        self.scale_dim = scale_dim
        self.shuffle = shuffle

        # Define flow architecture for layer
        xavier=tf.keras.initializers.GlorotUniform()
        
        self.mu_0 = tf.keras.layers.Dense(50,activation='relu')
        self.mu_1 = tf.keras.layers.Dense(50,activation='relu')
        self.mu_2 = tf.keras.layers.Dense(1,activation='linear')

        self.nu_0 = tf.keras.layers.Dense(50,activation='relu')
        self.nu_1 = tf.keras.layers.Dense(50,activation='relu')
        self.nu_2 = tf.keras.layers.Dense(1,activation='linear')

        if shuffle:
            self.idx = tf.random.shuffle(tf.range(num_dims))
            self.unran_idx = tf.argsort(self.idx)

    @tf.function
    def scale(self,x,params):
        z = tf.concat([x,params],axis=-1)

        z = self.mu_0(z)
        z = self.mu_1(z)
        z = self.mu_2(z)

        return z

    @tf.function
    def translate(self,x,params):
        z = tf.concat([x,params],axis=-1)

        z = self.nu_0(z)
        z = self.nu_1(z)
        z = self.nu_2(z)

        return z

    @tf.function
    def call(self,x,params):

        if self.shuffle:
            x = tf.transpose(tf.gather(tf.transpose(x),self.idx))

        zd = x[:,0:self.scale_dim]
        mu = self.scale(zd,params)

        zD = x[:,self.scale_dim:]*tf.exp(mu) + self.translate(zd,params)

        return tf.concat([zd,zD], axis =1), tf.reduce_sum(mu)

    @tf.function
    def invert(self,z,params):

        xd = z[:,0:self.scale_dim]
        mu = self.scale(xd,params)

        xD = (z[:,self.scale_dim:] - self.translate(xd,params))/(tf.exp(mu))

        x = tf.concat([xd,xD], axis =1)

        if self.shuffle:
            x = tf.transpose(tf.gather(tf.transpose(x),self.unran_idx))

        return x


""" 
Build a regular Real NVP flow layer
"""
class real_nvp(layers.Layer):
    def __init__(self,scale_dim,shuffle=False):
        super(real_nvp, self).__init__()

        self.scale_dim = scale_dim
        self.shuffle = shuffle

        # Define flow architecture for layer
        xavier=tf.keras.initializers.GlorotUniform()
        
        self.mu_0 = tf.keras.layers.Dense(50,activation='relu')
        self.mu_1 = tf.keras.layers.Dense(50,activation='relu')
        self.mu_2 = tf.keras.layers.Dense(1,activation='linear')

        self.nu_0 = tf.keras.layers.Dense(50,activation='relu')
        self.nu_1 = tf.keras.layers.Dense(50,activation='relu')
        self.nu_2 = tf.keras.layers.Dense(1,activation='linear')

    def build(self,input_shape): # Only needed when shuffling
        self.idx = tf.random.shuffle(tf.range(input_shape[-1]))
        self.unran_idx = tf.argsort(self.idx)

    @tf.function
    def scale(self,x):
        z = self.mu_0(x)
        z = self.mu_1(z)
        z = self.mu_2(z)

        return z

    @tf.function
    def translate(self,x):
        z = self.nu_0(x)
        z = self.nu_1(z)
        z = self.nu_2(z)

        return z

    @tf.function
    def call(self,x):

        if self.shuffle:
            x = tf.transpose(tf.gather(tf.transpose(x),self.idx))

        zd = x[:,0:self.scale_dim]
        mu = self.scale(zd)

        zD = x[:,self.scale_dim:]*tf.exp(mu) + self.translate(zd)

        return tf.concat([zd,zD], axis=1), tf.reduce_sum(mu)

    @tf.function
    def invert(self,z):
        xd = z[:,0:self.scale_dim]
        mu = self.scale(xd)

        xD = (z[:,self.scale_dim:] - self.translate(xd))/(tf.exp(mu))

        x = tf.concat([xd,xD], axis=1)

        if self.shuffle:
            x = tf.transpose(tf.gather(tf.transpose(x),self.unran_idx))

        return x



""" 
Build a neural spline flow layer - incomplete!!
"""
class neural_spline_flow(layers.Layer):
    def __init__(self,scale_dim,B,K=10,shuffle=False):
        super(real_nvp, self).__init__()

        self.scale_dim = scale_dim
        self.shuffle = shuffle
        self.B = B
        self.K = K


    def build(self,input_shape): # Only needed when shuffling
        self.idx = tf.random.shuffle(tf.range(input_shape[-1]))
        self.unran_idx = tf.argsort(self.idx)

        self.total_dim = input_shape[-1]
        self.transform_dim = self.total_dim-self.scale_dim

        # Define flow architecture for layer
        xavier=tf.keras.initializers.GlorotUniform()
        
        self.nsp_0 = tf.keras.layers.Dense(50,activation='tanh')
        self.nsp_1 = tf.keras.layers.Dense(50,activation='tanh')
        
        self.nsp_2_w = tf.keras.layers.Dense(self.transform_dim*self.K,activation='linear') # needs additional softmax
        self.nsp_2_h = tf.keras.layers.Dense(self.transform_dim*self.K,activation='linear') # needs additional softmax
        self.nsp_2_d = tf.keras.layers.Dense(self.transform_dim*(self.K-1),activation='linear') # needs additional softplus

    @tf.function
    def call(self,x):

        if self.shuffle:
            x = tf.transpose(tf.gather(tf.transpose(x),self.idx))

        total_dim = tf.shape(x)[-1] # Check syntax
        zd = x[:,0:self.scale_dim]

        hh_ = self.nsp_0(zd)
        hh_ = self.nsp_1(hh_)

        theta_w = 2.0*self.B*self.nsp_2_w(hh_)
        theta_w = tf.reshape(theta_w,[-1,self.transform_dim,self.K])
        theta_w = tf.nn.softmax(theta_w,axis=-1)

        theta_h = 2.0*self.B*self.nsp_2_h(hh_)
        theta_h = tf.reshape(theta_h,[-1,self.transform_dim,self.K])
        theta_h = tf.nn.softmax(theta_h,axis=-1)

        theta_d = self.nsp_2_d(hh_)
        theta_d = tf.reshape(theta_d,[-1,self.transform_dim,self.K-1])
        theta_d = tf.math.softplus(theta_d)

        binx_ = tf.math.cumsum(theta_w,axis=-1)
        biny_ = tf.math.cumsum(theta_h,axis=-1)


        return tf.concat([zd,zD], axis=1), tf.math.log(tf.abs(tf.exp(mu)))

    @tf.function
    def invert(self,z):
        xd = z[:,0:self.scale_dim]
        mu = self.scale(xd)

        xD = (z[:,self.scale_dim:] - self.translate(xd))/(tf.exp(mu))

        x = tf.concat([xd,xD], axis=1)

        if self.shuffle:
            x = tf.transpose(tf.gather(tf.transpose(x),self.unran_idx))

        return x

if __name__ == '__main__':
    print('Some testing')

    x = np.pi*tf.ones((1, 10),dtype='float64')
    my_layer_1 = real_nvp(scale_dim=2,shuffle=False)
    my_layer_2 = real_nvp(scale_dim=7,shuffle=True)
    
    y1, nll_y1 = my_layer_1(x)
    y2, nll_y2 = my_layer_2(y1)

    x_i = my_layer_2.invert(y2)
    x_i = my_layer_1.invert(x_i)

    print(x_i) # This should be very close to x