import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

import tensorflow as tf
tf.random.set_seed(10)

from tensorflow.keras import Model
import numpy as np
np.random.seed(10)

from utils import coeff_determination
from sklearn.preprocessing import MinMaxScaler

#Build the model which does basic map of inputs to coefficients
class coefficient_model(Model):
    def __init__(self,input_data,output_data):
        super(coefficient_model, self).__init__()

        # Scale
        self.op_scaler = MinMaxScaler()
        self.op_scaler.fit_transform(output_data)

        # Randomize datasets
        idx = np.arange(np.shape(input_data)[0])
        np.random.shuffle(idx)

        # Segregate
        self.input_data_train = input_data[idx[:140]]
        self.output_data_train = output_data[idx[:140]]

        self.input_data_test = input_data[idx[140:]]
        self.output_data_test = output_data[idx[140:]]

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
        r2_iter = 0
        for i in range(200):
            for batch in range(7):
                input_batch = self.input_data_train[batch*20:(batch+1)*20]
                output_batch = self.output_data_train[batch*20:(batch+1)*20]
                self.network_learn(input_batch,output_batch)

                
            # Check accuracy
            if r2_iter == 10:
                predictions = self.call(self.input_data_test)
                print('Test loss:',self.get_loss(self.input_data_test,self.output_data_test).numpy())
                print('Test loss:',self.get_loss(self.input_data_test,self.output_data_test))
                r2 = coeff_determination(predictions,self.output_data_test)
                print('Test R2:',r2)
                r2_iter = 0

                # Save the model every 100 iterations
                self.save_weights('./checkpoints/my_checkpoint')
            else:
                r2_iter = r2_iter + 1

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

'''
Next few model classes are still work in progress
'''
#Build the model which does basic map of inputs to coefficients but through a constrained linear projection (POD)
class projection_model(Model):
    def __init__(self,input_data,field_data,coeff_data,omat):
        super(coefficient_model, self).__init__()

        # Randomize datasets
        idx = np.arange(np.shape(input_data)[0])
        np.random.shuffle(idx)

        # Segregate
        self.input_data_train = input_data[idx[:140]]
        self.input_data_test = input_data[idx[140:]]
        self.ip_shape = np.shape(input_data)[1]

        self.coeff_data_train = coeff_data[idx[:140]]
        self.coeff_data_test = coeff_data[idx[140:]]

        # Training snapshots
        field_train = field_data[:,idx[:140]]
        # Get rid of snapshot matrix mean
        sm_mean = np.mean(field_train,axis=1)
        field_train = field_train[:,:] - sm_mean[:,None]
        # Testing snapshots
        field_test = field_data[:,idx[140:]]
        field_test = field_test[:,:] - sm_mean[:,None]
        # Perform POD
        new_mat = np.matmul(np.transpose(field_train),field_train)
        w,v = LA.eig(new_mat)
        # Bases
        phi = np.real(np.matmul(field_train,v))
        trange = np.arange(140)
        phi[:,trange] = phi[:,trange]/np.sqrt(w[:])

        output_data_train = np.matmul(np.transpose(phi),field_train)
        output_data_test = np.matmul(np.transpose(phi),field_test)

        self.op_shape = np.shape(output_data_train)[1]
        self.basis = tf.Variable(phi.astype('float32'))

        # Observation matrix to get coefficients
        self.omat = tf.Variable(mat.astype('float32'))

        # Define model
        xavier=tf.keras.initializers.GlorotUniform()
        self.l1=tf.keras.layers.Dense(10,kernel_initializer=xavier,activation=tf.nn.relu,input_shape=[self.ip_shape])
        self.l2=tf.keras.layers.Dense(25,kernel_initializer=xavier,activation=tf.nn.relu)
        self.l3=tf.keras.layers.Dense(50,kernel_initializer=xavier,activation=tf.nn.relu)
        self.l4=tf.keras.layers.Dense(75,kernel_initializer=xavier,activation=tf.nn.relu)
        self.l5=tf.keras.layers.Dense(50,kernel_initializer=xavier,activation=tf.nn.relu)
        self.l6=tf.keras.layers.Dense(25,kernel_initializer=xavier,activation=tf.nn.relu)
        self.l7=tf.keras.layers.Dense(10,kernel_initializer=xavier,activation=tf.nn.relu)
        self.out=tf.keras.layers.Dense(self.op_shape,kernel_initializer=xavier)
        self.train_op = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Running the model
    def call(self,X):
        boom=self.l1(X)
        boom=self.l2(boom)
        boom=self.l3(boom)
        boom=self.l4(boom)
        boom=self.l5(boom)
        boom=self.l6(boom)
        boom=self.l7(boom)
        boom=self.out(boom)
        return boom
    
    # Regular MSE
    def get_loss(self,X,Y,C): # The Y are POD coefficients and the C are lift/drag coefficients
        boom=self.call(X) # Predicted coefficients
        field = tf.linalg.matmul(boom,self.basis) # The field is reconstruct as a tf variable
        cboom = tf.linalg.matmul(self.omat,field)

        return tf.reduce_mean(tf.math.square(boom-Y)) + tf.reduce_mean(tf.math.square(cboom-C))

    # get gradients - regular
    def get_grad(self,X,Y,C):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(X,Y,C)
            g = tape.gradient(L, self.trainable_variables)
        return g
    
    # perform gradient descent - regular
    def network_learn(self,X,Y,C):
        g = self.get_grad(X,Y,C)
        self.train_op.apply_gradients(zip(g, self.trainable_variables))

    # Train the model
    def train_model(self):
        plot_iter = 0
        r2_iter = 0
        for i in range(100):
            for batch in range(7):
                input_batch = self.input_data_train[batch*20:(batch+1)*20]
                output_batch = self.output_data_train[batch*20:(batch+1)*20]
                coeff_batch = self.coeff_data_train[batch*20:(batch+1)*20]
                self.network_learn(input_batch,output_batch,coeff_batch)

            # Check accuracy
            if r2_iter == 10:
                print('Test loss:',self.get_loss(self.input_data_test,self.output_data_test,self.coeff_data_test))
                predictions = self.call(self.input_data_test)
                r2 = coeff_determination(predictions,self.output_data_test)
                print('Test R2:',r2)
                r2_iter = 0

                # Save the model every 100 iterations
                self.save_weights('./checkpoints/my_checkpoint')
            else:
                r2_iter = r2_iter + 1

    # Load weights
    def restore_model(self):
        self.load_weights(dir_path+'/checkpoints/my_checkpoint') # Load pretrained model



# Build the model which does basic map of inputs to coefficients but through a constrained linear projection (POD)
# Also enhance by adjoint information
class projection_model_adjoint(Model):
    def __init__(self,input_data,field_data,coeff_data,omat,adjoint_data):
        super(coefficient_model, self).__init__()

        # Randomize datasets
        idx = np.arange(np.shape(input_data)[0])
        np.random.shuffle(idx)

        # Segregate
        self.input_data_train = input_data[idx[:140]]
        self.input_data_test = input_data[idx[140:]]
        self.ip_shape = np.shape(input_data)[1]

        self.coeff_data_train = coeff_data[idx[:140]]
        self.coeff_data_test = coeff_data[idx[140:]]

        self.adjoint_data_train = adjoint_data[idx[:140]]
        self.adjoint_data_test = adjoint_data[idx[140:]]

        # Training snapshots
        field_train = field_data[:,idx[:140]]
        # Get rid of snapshot matrix mean
        sm_mean = np.mean(field_train,axis=1)
        field_train = field_train[:,:] - sm_mean[:,None]
        # Testing snapshots
        field_test = field_data[:,idx[140:]]
        field_test = field_test[:,:] - sm_mean[:,None]
        # Perform POD
        new_mat = np.matmul(np.transpose(field_train),field_train)
        w,v = LA.eig(new_mat)
        # Bases
        phi = np.real(np.matmul(field_train,v))
        trange = np.arange(140)
        phi[:,trange] = phi[:,trange]/np.sqrt(w[:])

        output_data_train = np.matmul(np.transpose(phi),field_train)
        output_data_test = np.matmul(np.transpose(phi),field_test)

        self.op_shape = np.shape(output_data_train)[1]
        self.basis = tf.Variable(phi.astype('float32'))

        # Observation matrix to get coefficients
        self.omat = tf.Variable(mat.astype('float32'))

        # Define model
        xavier=tf.keras.initializers.GlorotUniform()
        self.l1=tf.keras.layers.Dense(10,kernel_initializer=xavier,activation=tf.nn.relu,input_shape=[self.ip_shape])
        self.l2=tf.keras.layers.Dense(25,kernel_initializer=xavier,activation=tf.nn.relu)
        self.l3=tf.keras.layers.Dense(50,kernel_initializer=xavier,activation=tf.nn.relu)
        self.l4=tf.keras.layers.Dense(75,kernel_initializer=xavier,activation=tf.nn.relu)
        self.l5=tf.keras.layers.Dense(50,kernel_initializer=xavier,activation=tf.nn.relu)
        self.l6=tf.keras.layers.Dense(25,kernel_initializer=xavier,activation=tf.nn.relu)
        self.l7=tf.keras.layers.Dense(10,kernel_initializer=xavier,activation=tf.nn.relu)
        self.out=tf.keras.layers.Dense(self.op_shape,kernel_initializer=xavier)
        self.train_op = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Running the model
    def call(self,X):
        boom=self.l1(X)
        boom=self.l2(boom)
        boom=self.l3(boom)
        boom=self.l4(boom)
        boom=self.l5(boom)
        boom=self.l6(boom)
        boom=self.l7(boom)
        boom=self.out(boom)
        return boom
    
    # Regular MSE
    def get_loss(self,X,Y,C,A): # The Y are POD coefficients and the C are lift/drag coefficients, A is adjoint information
        boom=self.call(X) # Predicted coefficients
        field = tf.linalg.matmul(boom,self.basis) # The field is reconstruct as a tf variable
        cboom = tf.linalg.matmul(self.omat,field)

        return tf.reduce_mean(tf.math.square(boom-Y)) + tf.reduce_mean(tf.math.square(cboom-C))

    # get gradients - regular
    def get_grad(self,X,Y,C,A):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(X,Y,C)
            g = tape.gradient(L, self.trainable_variables)
        return g
    
    # perform gradient descent - regular
    def network_learn(self,X,Y,C,A):
        g = self.get_grad(X,Y,C)
        self.train_op.apply_gradients(zip(g, self.trainable_variables))

    # Train the model
    def train_model(self):
        plot_iter = 0
        r2_iter = 0
        for i in range(100):
            for batch in range(7):
                input_batch = self.input_data_train[batch*20:(batch+1)*20]
                output_batch = self.output_data_train[batch*20:(batch+1)*20]
                coeff_batch = self.coeff_data_train[batch*20:(batch+1)*20]
                adjoint_batch = self.adjoint_data_train[batch*20:(batch+1)*20]
                self.network_learn(input_batch,output_batch,coeff_batch,adjoint_batch)

            # Check accuracy
            if r2_iter == 10:
                print('Test loss:',self.get_loss(self.input_data_test,self.output_data_test,self.coeff_data_test,self.adjoint_data_test))
                predictions = self.call(self.input_data_test)
                r2 = coeff_determination(predictions,self.output_data_test)
                print('Test R2:',r2)
                r2_iter = 0

                # Save the model every 100 iterations
                self.save_weights('./checkpoints/my_checkpoint')
            else:
                r2_iter = r2_iter + 1

    # Load weights
    def restore_model(self):
        self.load_weights(dir_path+'/checkpoints/my_checkpoint') # Load pretrained model


if __name__ == '__main__':
    pass