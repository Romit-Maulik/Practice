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
    def __init__(self, scale_dim, shuffle=False):
        super(parametric_real_nvp, self).__init__()

        self.scale_dim = scale_dim
        self.shuffle = shuffle

        # Define flow architecture for layer
        xavier=tf.keras.initializers.GlorotUniform()
        
        self.mu_0 = tf.keras.layers.Dense(50,activation='sigmoid')
        self.mu_1 = tf.keras.layers.Dense(50,activation='sigmoid')
        self.mu_2 = tf.keras.layers.Dense(1,activation='sigmoid')

        self.nu_0 = tf.keras.layers.Dense(50,activation='sigmoid')
        self.nu_1 = tf.keras.layers.Dense(50,activation='sigmoid')
        self.nu_2 = tf.keras.layers.Dense(1,activation='sigmoid')

    def build(self,input_shape): # Only needed when shuffling
        self.idx = tf.random.shuffle(tf.range(input_shape[-1]))
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

        return tf.concat([zd,zD], axis =1), tf.math.log(tf.abs(tf.exp(mu)))

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
Build a scalar parametric Real NVP flow layer
"""
class scalar_real_nvp(layers.Layer):
    def __init__(self):
        super(scalar_real_nvp, self).__init__()

        # Define flow architecture for layer
        xavier=tf.keras.initializers.GlorotUniform()
        
        self.mu_0 = tf.keras.layers.Dense(50,activation='sigmoid')
        self.mu_1 = tf.keras.layers.Dense(50,activation='sigmoid')
        self.mu_2 = tf.keras.layers.Dense(1,activation='sigmoid')

        self.nu_0 = tf.keras.layers.Dense(50,activation='sigmoid')
        self.nu_1 = tf.keras.layers.Dense(50,activation='sigmoid')
        self.nu_2 = tf.keras.layers.Dense(1,activation='sigmoid')

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
    def call(self,x,params):

        mu = self.scale(params)
        z = x*tf.exp(mu) + self.translate(params)

        return z, tf.math.log(tf.abs(tf.exp(mu)))

    @tf.function
    def invert(self,z,params):
        
        mu = self.scale(params)
        x = (z - self.translate(params))/(tf.exp(mu))
        
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
        
        self.mu_0 = tf.keras.layers.Dense(50,activation='sigmoid')
        self.mu_1 = tf.keras.layers.Dense(50,activation='sigmoid')
        self.mu_2 = tf.keras.layers.Dense(1,activation='sigmoid')

        self.nu_0 = tf.keras.layers.Dense(50,activation='sigmoid')
        self.nu_1 = tf.keras.layers.Dense(50,activation='sigmoid')
        self.nu_2 = tf.keras.layers.Dense(1,activation='sigmoid')

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