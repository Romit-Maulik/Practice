import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

import tensorflow as tf
tf.random.set_seed(10)

from tensorflow.keras import Model
import numpy as np
np.random.seed(10)

from utils import coeff_determination
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Build the model which does basic map of inputs to coefficients
class coefficient_model(Model):
    def __init__(self,input_data,output_data):
        super(coefficient_model, self).__init__()

        # Scale
        # self.op_scaler = StandardScaler()
        self.op_scaler = MinMaxScaler()
        self.op_scaler.fit_transform(output_data)

        # Randomize datasets
        idx = np.arange(np.shape(input_data)[0])
        np.random.shuffle(idx)

        # Segregate
        self.input_data_train = input_data[idx[:50]]
        self.output_data_train = output_data[idx[:50]]

        self.input_data_valid = input_data[idx[50:100]]
        self.output_data_valid = output_data[idx[50:100]]

        self.input_data_test = input_data[idx[100:]]
        self.output_data_test = output_data[idx[100:]]

        self.ip_shape = np.shape(input_data)[1]
        self.op_shape = np.shape(output_data)[1]

        # Define model
        xavier=tf.keras.initializers.GlorotUniform()
        self.l1=tf.keras.layers.Dense(100,kernel_initializer=xavier,activation=tf.nn.relu,input_shape=[self.ip_shape])
        self.l2=tf.keras.layers.Dense(100,kernel_initializer=xavier,activation=tf.nn.relu)
        self.l3=tf.keras.layers.Dense(100,kernel_initializer=xavier,activation=tf.nn.relu)
        self.out=tf.keras.layers.Dense(self.op_shape,kernel_initializer=xavier,activation=tf.nn.relu)
        self.train_op = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Running the model
    def call(self,X):
        boom=self.l1(X)
        boom=self.l2(boom)
        boom=self.l3(boom)
        boom=self.out(boom)
        return boom
    
    # Regular MSE
    def get_loss(self,X,Y):
        boom=self.call(X)
        return tf.reduce_mean(tf.math.square(boom-Y))

    # get gradients - regular
    def get_grad(self,X,Y):
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
        patience = 50
        best_valid_loss = 999999.0 # Some large number 
        
        for i in range(2000):
            # Training loss
            for batch in range(10):
                input_batch = self.input_data_train[batch*5:(batch+1)*5]
                output_batch = self.output_data_train[batch*5:(batch+1)*5]
                self.network_learn(input_batch,output_batch)

            # Validation loss
            valid_loss = 0.0
            valid_r2 = 0.0
            for batch in range(10):
                input_batch = self.input_data_valid[batch*5:(batch+1)*5]
                output_batch = self.output_data_valid[batch*5:(batch+1)*5]
                valid_loss = valid_loss + self.get_loss(self.input_data_valid,self.output_data_valid).numpy()

                predictions = self.call(self.input_data_valid)
                valid_r2 = valid_r2 + coeff_determination(predictions,self.output_data_valid)

            valid_r2 = valid_r2/(batch+1)
                

            # Check early stopping criteria
            if valid_loss < best_valid_loss:
                
                print('Improved validation loss from:',best_valid_loss,' to:', valid_loss)
                print('Validation R2:',valid_r2)
                
                best_valid_loss = valid_loss
                self.save_weights('./checkpoints/my_checkpoint')
                stop_iter = 0
            else:
                stop_iter = stop_iter + 1

            if stop_iter == patience:
                break
                
        # Check accuracy on test
        predictions = self.call(self.input_data_test)
        print('Test loss:',self.get_loss(self.input_data_test,self.output_data_test).numpy())
        r2 = coeff_determination(predictions,self.output_data_test)
        print('Test R2:',r2)
        r2_iter = 0            

    # Load weights
    def restore_model(self):
        self.load_weights(dir_path+'/checkpoints/my_checkpoint') # Load pretrained model


#Build the model which predicts coefficients enhanced by adjoint information based loss
class coefficient_model_adjoint(Model):
    def __init__(self,input_data,output_data,adjoint_data):
        super(coefficient_model_adjoint, self).__init__()

        # Scale
        self.op_scaler = MinMaxScaler()
        self.op_scaler.fit_transform(output_data)

        # Randomize
        idx = np.arange(np.shape(input_data)[0])
        np.random.shuffle(idx)

        # Segregate
        self.input_data_train = input_data[idx[:140]]
        self.output_data_train = output_data[idx[:140]]
        self.adjoint_data_train = adjoint_data[idx[:140]]

        self.input_data_test = input_data[idx[140:]]
        self.output_data_test = output_data[idx[140:]]
        self.adjoint_data_test = adjoint_data[idx[140:]]

        # Shapes
        self.ip_shape = np.shape(input_data)[1]
        self.op_shape = np.shape(output_data)[1]

        # Define model
        xavier=tf.keras.initializers.GlorotUniform()
        self.l1=tf.keras.layers.Dense(20,kernel_initializer=xavier,activation=tf.nn.tanh,input_shape=[self.ip_shape])
        self.l2=tf.keras.layers.Dense(20,kernel_initializer=xavier,activation=tf.nn.tanh)
        self.l3=tf.keras.layers.Dense(20,kernel_initializer=xavier,activation=tf.nn.tanh)
        self.out=tf.keras.layers.Dense(self.op_shape,kernel_initializer=xavier,activation=tf.nn.tanh)
        self.train_op = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Running the model
    def call(self,X):
        boom=self.l1(X)
        boom=self.l2(boom)
        boom=self.l3(boom)
        boom=self.out(boom)
        return boom
    
    # Adjoint enhanced MSE
    def get_loss(self,X,Y,A):
        input_var = tf.Variable(X.astype('float32'))
        with tf.GradientTape() as tape:
            tape.watch(input_var)
            boom=self.call(input_var)
            op = tf.math.square(boom)[0][0]
            g = tf.reduce_sum(tf.square(tape.gradient(op,input_var)-A))

        return tf.reduce_mean(tf.math.square(boom-Y)) + g

    # get gradients - adjoint enhanced
    def get_grad(self,X,Y,A):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(X,Y,A)
            g = tape.gradient(L, self.trainable_variables)
        return g

    # perform gradient descent - adjoint enhanced
    def network_learn(self,X,Y,A):
        g = self.get_grad(X,Y,A)
        self.train_op.apply_gradients(zip(g, self.trainable_variables))

    # Train the model
    def train_model(self):
        plot_iter = 0
        r2_iter = 0
        for i in range(200):
            for batch in range(7):
                input_batch = self.input_data_train[batch*20:(batch+1)*20]
                output_batch = self.output_data_train[batch*20:(batch+1)*20]
                adjoint_batch = self.adjoint_data_train[batch*20:(batch+1)*20]
                
                self.network_learn(input_batch,output_batch,adjoint_batch)

            # Check accuracy
            if r2_iter == 10:
                print('Test loss:',self.get_loss(self.input_data_test,self.output_data_test,self.adjoint_data_test))
                predictions = self.call(self.input_data_test)
                r2 = coeff_determination(predictions,self.output_data_test)
                print('Test R2:',r2)
                r2_iter = 0
                self.save_weights('./checkpoints/my_checkpoint')
            else:
                r2_iter = r2_iter + 1

    # Load weights
    def restore_model(self):
        self.load_weights(dir_path+'/checkpoints/my_checkpoint') # Load pretrained model

if __name__ == '__main__':
    # Load dataset
    input_data = np.load('doe_data.npy').astype('float32')
    output_data = np.load('coeff_data.npy').astype('float32')
    # Define a simple fully connected model
    model=coefficient_model(input_data,output_data)
    # Restore
    model.restore_model()
    # Predict
    inputs = np.asarray([0.1009, 0.3306, 0.6281, 0.1494, -0.1627, -0.6344, -0.5927, 0.0421]).astype('float32')
    inputs = inputs.reshape(1,8)
    pred = model.predict(inputs)
    print(pred)