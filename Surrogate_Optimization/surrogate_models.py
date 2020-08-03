import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

import tensorflow as tf
tf.random.set_seed(10)

from tensorflow.keras import Model
import numpy as np
np.random.seed(10)

from utils import coeff_determination, plot_scatter, plot_stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Build the model which does basic map of inputs to coefficients
class coefficient_model(Model):
    def __init__(self,input_data,output_data):
        super(coefficient_model, self).__init__()

        # Scale
        # self.op_scaler = StandardScaler()
        self.op_scaler = MinMaxScaler()
        output_data = self.op_scaler.fit_transform(output_data)

        # Randomize datasets
        idx = np.arange(np.shape(input_data)[0])
        np.random.shuffle(idx)

        self.ntrain = 30
        self.nvalid = 30

        # Segregate
        self.input_data_train = input_data[idx[:self.ntrain]]
        self.output_data_train = output_data[idx[:self.ntrain]]

        self.input_data_valid = input_data[idx[self.ntrain:self.ntrain+self.nvalid]]
        self.output_data_valid = output_data[idx[self.ntrain:self.ntrain+self.nvalid]]

        self.input_data_test = input_data[idx[self.ntrain+self.nvalid:]]
        self.output_data_test = output_data[idx[self.ntrain+self.nvalid:]]

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

    #     # Adjoint enhanced MSE
    # def get_loss(self,X,Y,A):
    #     input_var = tf.Variable(X.astype('float32'))
    #     with tf.GradientTape() as tape:
    #         tape.watch(input_var)
    #         boom=self.call(input_var)
    #         op = tf.math.square(boom)[0][0]
    #         g = tf.reduce_sum(tf.square(tape.gradient(op,input_var)-A))

    #     return tf.reduce_mean(tf.math.square(boom-Y)) + g

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

        self.num_batches = 10
        self.train_batch_size = int(self.ntrain/self.num_batches)
        self.valid_batch_size = int(self.nvalid/self.num_batches)

        f = open("training_stats.csv","w")
        f.write('Train loss, Validation loss, Train R2, Validation R2'+'\n')
        f.close()
        
        for i in range(2000):
            # Training loss
            train_loss = 0.0
            train_r2 = 0.0
            for batch in range(self.num_batches):
                input_batch = self.input_data_train[batch*self.train_batch_size:(batch+1)*self.train_batch_size]
                output_batch = self.output_data_train[batch*self.train_batch_size:(batch+1)*self.train_batch_size]
                self.network_learn(input_batch,output_batch)

                train_loss = train_loss + self.get_loss(input_batch,output_batch).numpy()
                predictions = self.call(input_batch)
                train_r2 = train_r2 + coeff_determination(predictions,output_batch)

            train_r2 = train_r2/(batch+1)

            # Validation loss
            valid_loss = 0.0
            valid_r2 = 0.0
            for batch in range(self.num_batches):
                input_batch = self.input_data_valid[batch*self.valid_batch_size:(batch+1)*self.valid_batch_size]
                output_batch = self.output_data_valid[batch*self.valid_batch_size:(batch+1)*self.valid_batch_size]
                
                valid_loss = valid_loss + self.get_loss(input_batch,output_batch).numpy()
                predictions = self.call(input_batch)
                valid_r2 = valid_r2 + coeff_determination(predictions,output_batch)

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

            # Write metrics to file
            f = open("training_stats.csv","a")
            f.write(str(train_loss)+','+str(valid_loss)+','+str(train_r2)+','+str(valid_r2)+'\n')
            f.close()

            if stop_iter == patience:
                break
                
        # Check accuracy on test
        predictions = self.call(self.input_data_test)
        print('Test loss:',self.get_loss(self.input_data_test,self.output_data_test).numpy())
        r2 = coeff_determination(predictions,self.output_data_test)
        print('Test R2:',r2)
        r2_iter = 0

        # Plot scatter on the test
        predictions = self.op_scaler.inverse_transform(predictions)
        truth = self.op_scaler.inverse_transform(self.output_data_test)
        plot_scatter(predictions[:,0],truth[:,0],'Drag')
        plot_scatter(predictions[:,1],truth[:,1],'Lift')

        # Plot training stats
        plot_stats()

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