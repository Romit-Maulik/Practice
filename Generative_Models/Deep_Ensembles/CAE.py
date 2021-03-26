import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set seeds
np.random.seed(10)
tf.random.set_seed(10)

from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Dropout
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import Model
from tensorflow.keras import backend as K
tf.keras.backend.set_floatx('float32')

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Custom activation (swish)
def my_swish(x, beta=1.0):
    return x * K.sigmoid(beta * x)

class cnn_autoencoder(Model):
    def __init__(self,data_tuple,arch_type='baseline',eps_range=0.01):
        super(cnn_autoencoder, self).__init__()

        self.train_data = data_tuple[0]
        self.valid_data = data_tuple[1]
        self.test_data = data_tuple[2]

        self.ntrain = self.train_data.shape[0]
        self.nvalid = self.valid_data.shape[0]
        self.arch_type = arch_type
        self.num_latent = 6

        self.init_architecture_baseline()
        self.train_op = tf.keras.optimizers.Adam(learning_rate=0.001)

        # If adversarial training is used
        self.eps_range = eps_range

    def init_architecture_baseline(self):

        # Define model architecture
        ## Encoder
        self.enc_l1 = Conv2D(30,kernel_size=(3,3),activation=my_swish,padding='same')
        self.enc_l2 = MaxPooling2D(pool_size=(2, 2),padding='same')

        self.enc_l3 = Conv2D(25,kernel_size=(3,3),activation=my_swish,padding='same')
        self.enc_l4 = MaxPooling2D(pool_size=(2, 2),padding='same')

        self.enc_l5 = Conv2D(20,kernel_size=(3,3),activation=my_swish,padding='same')
        self.enc_l6 = MaxPooling2D(pool_size=(2, 2),padding='same')

        self.enc_l7 = Conv2D(15,kernel_size=(3,3),activation=my_swish,padding='same')
        self.enc_l8 = MaxPooling2D(pool_size=(2, 2),padding='same')

        self.enc_l9 = Conv2D(10,kernel_size=(3,3),activation=my_swish,padding='same')
        self.enc_l10 = MaxPooling2D(pool_size=(2, 2),padding='same')

        self.enc_l11 = Flatten()
        self.enc_l12 = Dense(50, activation=my_swish)
        self.enc_l13 = Dense(25, activation=my_swish)
        self.enc_l14 = Dense(10, activation=my_swish)

        self.encoded = Dense(self.num_latent)

        ## Decoder
        self.dec_l1 = Dense(10,activation=my_swish)
        self.dec_l2 = Dense(25,activation=my_swish)
        self.dec_l3 = Dense(50,activation=my_swish)
        self.dec_l4 = Dense(2*2*3,activation=my_swish)

        self.dec_l5 = Reshape(target_shape=(2,2,3))

        self.dec_l6 = Conv2D(10,kernel_size=(3,3),activation=my_swish,padding='same')
        self.dec_l7 = UpSampling2D(size=(2, 2))

        self.dec_l8 = Conv2D(15,kernel_size=(3,3),activation=my_swish,padding='same')
        self.dec_l9 = UpSampling2D(size=(2, 2))

        self.dec_l10 = Conv2D(20,kernel_size=(3,3),activation=my_swish,padding='same')
        self.dec_l11 = UpSampling2D(size=(2, 2))

        self.dec_l12 = Conv2D(25,kernel_size=(3,3),activation=my_swish,padding='same')
        self.dec_l13 = UpSampling2D(size=(2, 2))

        self.dec_l14 = Conv2D(30,kernel_size=(3,3),activation=my_swish,padding='same')
        self.dec_l15 = UpSampling2D(size=(2, 2))

        if self.arch_type == 'baseline':
            self.decoded = Conv2D(1,kernel_size=(3,3),activation='linear',padding='same')
        
        elif self.arch_type =='mixture' or self.arch_type == 'ensemble':
            self.decoded_mean = Conv2D(1,kernel_size=(3,3),activation='linear',padding='same')
            self.decoded_logvar = Conv2D(1,kernel_size=(3,3),activation='linear',padding='same') #log(sigma^2)

        if self.arch_type == 'dropout':
            self.decoded = Conv2D(1,kernel_size=(3,3),activation='linear',padding='same')
            self.dropout_layer = Dropout(0.1)

    def call_baseline(self,X):

        # Encode
        hh = self.enc_l1(X)
        hh = self.enc_l2(hh)
        hh = self.enc_l3(hh)
        hh = self.enc_l4(hh)
        hh = self.enc_l5(hh)
        hh = self.enc_l6(hh)
        hh = self.enc_l7(hh)
        hh = self.enc_l8(hh)
        hh = self.enc_l9(hh)
        hh = self.enc_l10(hh)
        hh = self.enc_l11(hh)
        hh = self.enc_l12(hh)
        hh = self.enc_l13(hh)
        hh = self.enc_l14(hh)

        latent = self.encoded(hh)

        # decode
        hh = self.dec_l1(latent)
        hh = self.dec_l2(hh)
        hh = self.dec_l3(hh)
        hh = self.dec_l4(hh)
        hh = self.dec_l5(hh)
        hh = self.dec_l6(hh)
        hh = self.dec_l7(hh)
        hh = self.dec_l8(hh)
        hh = self.dec_l9(hh)
        hh = self.dec_l10(hh)
        hh = self.dec_l11(hh)
        hh = self.dec_l12(hh)
        hh = self.dec_l13(hh)
        hh = self.dec_l14(hh)
        hh = self.dec_l15(hh)
        
        reconstructed = self.decoded(hh)

        return reconstructed, latent

    def call_dropout(self,X):

        # Encode
        hh = self.enc_l1(X)
        hh = self.dropout_layer(hh,training=True)
        hh = self.enc_l2(hh)
        
        hh = self.enc_l3(hh)
        hh = self.dropout_layer(hh,training=True)
        hh = self.enc_l4(hh)

        hh = self.enc_l5(hh)
        hh = self.dropout_layer(hh,training=True)
        hh = self.enc_l6(hh)

        hh = self.enc_l7(hh)
        hh = self.dropout_layer(hh,training=True)
        hh = self.enc_l8(hh)

        hh = self.enc_l9(hh)
        hh = self.dropout_layer(hh,training=True)
        hh = self.enc_l10(hh)
        hh = self.enc_l11(hh)

        hh = self.enc_l12(hh)
        hh = self.dropout_layer(hh,training=True)
        hh = self.enc_l13(hh)
        hh = self.dropout_layer(hh,training=True)
        hh = self.enc_l14(hh)
        hh = self.dropout_layer(hh,training=True)

        latent = self.encoded(hh)

        # decode
        hh = self.dec_l1(latent)
        hh = self.dropout_layer(hh,training=True)

        hh = self.dec_l2(hh)
        hh = self.dropout_layer(hh,training=True)
        
        hh = self.dec_l3(hh)
        hh = self.dropout_layer(hh,training=True)
        
        hh = self.dec_l4(hh)
        hh = self.dropout_layer(hh,training=True)
        
        hh = self.dec_l5(hh)
        hh = self.dec_l6(hh)
        hh = self.dropout_layer(hh,training=True)

        hh = self.dec_l7(hh)
        hh = self.dec_l8(hh)
        hh = self.dropout_layer(hh,training=True)
        hh = self.dec_l9(hh)

        hh = self.dec_l10(hh)
        hh = self.dropout_layer(hh,training=True)

        hh = self.dec_l11(hh)
        hh = self.dec_l12(hh)
        hh = self.dropout_layer(hh,training=True)
        
        hh = self.dec_l13(hh)
        hh = self.dec_l14(hh)
        hh = self.dropout_layer(hh,training=True)
        
        hh = self.dec_l15(hh)
        
        reconstructed = self.decoded(hh)

        return reconstructed, latent
    
    def call_mixture(self,X):

        # Encode
        hh = self.enc_l1(X)
        hh = self.enc_l2(hh)
        hh = self.enc_l3(hh)
        hh = self.enc_l4(hh)
        hh = self.enc_l5(hh)
        hh = self.enc_l6(hh)
        hh = self.enc_l7(hh)
        hh = self.enc_l8(hh)
        hh = self.enc_l9(hh)
        hh = self.enc_l10(hh)
        hh = self.enc_l11(hh)
        hh = self.enc_l12(hh)
        hh = self.enc_l13(hh)
        hh = self.enc_l14(hh)

        latent = self.encoded(hh)

        # decode
        hh = self.dec_l1(latent)
        hh = self.dec_l2(hh)
        hh = self.dec_l3(hh)
        hh = self.dec_l4(hh)
        hh = self.dec_l5(hh)
        hh = self.dec_l6(hh)
        hh = self.dec_l7(hh)
        hh = self.dec_l8(hh)
        hh = self.dec_l9(hh)
        hh = self.dec_l10(hh)
        hh = self.dec_l11(hh)
        hh = self.dec_l12(hh)
        hh = self.dec_l13(hh)
        hh = self.dec_l14(hh)
        hh = self.dec_l15(hh)
        
        reconstructed_mean = self.decoded_mean(hh)
        reconstructed_logvar = self.decoded_logvar(hh)

        return reconstructed_mean, reconstructed_logvar, latent

    # Running the model
    def call(self,X):
        if self.arch_type == 'baseline':
            recon, latent = self.call_baseline(X)
            return recon, latent
        elif self.arch_type =='dropout':
            recon, latent = self.call_dropout(X)
            return recon, latent
        elif self.arch_type =='mixture' or self.arch_type == 'ensemble':
            recon_mean, recon_logvar, latent = self.call_mixture(X)
            return recon_mean, recon_logvar, latent
    
    # Regular MSE
    def get_loss(self,X):

        if self.arch_type == 'mixture' or self.arch_type == 'ensemble': # Log likelihood optimization
            op_mean, op_logvar, _ = self.call(X)

            half_logvar = 0.5*op_logvar
            op_var = tf.math.exp(op_logvar)

            mse = (tf.math.square(op_mean-X))*0.5/(op_var+K.epsilon())
            loss = tf.reduce_mean(half_logvar+mse)

        else: 
            
            op, _ = self.call(X)
            loss = tf.reduce_mean(tf.math.square(op-X))

        return loss

    # Regular MSE
    def get_adversarial_loss(self,X,X_adv):

        op_mean, op_logvar, _ = self.call(X_adv)

        half_logvar = 0.5*op_logvar
        op_var = tf.math.exp(op_logvar)

        mse = tf.math.square(op_mean-X)*0.5/(op_var+K.epsilon())
        loss = tf.reduce_mean(half_logvar+mse)

        return loss

    # get gradients - regular
    def get_grad(self,X):
        if self.arch_type == 'ensemble':
            # Adversarial training
            X = tf.convert_to_tensor(X)
            with tf.GradientTape() as tape:
                tape.watch(X)
                L = self.get_loss(X)
                g_temp = tape.gradient(L,X)

            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                X_adv = X + self.eps_range*tf.math.sign(g_temp)
                L_adv = self.get_adversarial_loss(X,X_adv)
                g = tape.gradient(L+L_adv, self.trainable_variables)
        else:
            # Regular training
            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                L = self.get_loss(X)
                g = tape.gradient(L, self.trainable_variables)            
        
        return g
    
    # perform gradient descent - regular
    def network_learn(self,X):
        g = self.get_grad(X)
        self.train_op.apply_gradients(zip(g, self.trainable_variables))

    # Train the model
    def train_model(self):
        plot_iter = 0
        stop_iter = 0
        patience = 10
        best_valid_loss = np.inf # Some large number 

        self.num_batches = 12
        self.train_batch_size = int(self.ntrain/self.num_batches)
        self.valid_batch_size = int(self.nvalid/self.num_batches)
        
        for i in range(1000):
            # Training loss
            print('Training iteration:',i)
            
            for batch in range(self.num_batches):
                input_batch = self.train_data[batch*self.train_batch_size:(batch+1)*self.train_batch_size]
                self.network_learn(input_batch)

            # Validation loss
            valid_loss = 0.0

            for batch in range(self.num_batches):
                input_batch = self.valid_data[batch*self.valid_batch_size:(batch+1)*self.valid_batch_size]
                valid_loss = valid_loss + self.get_loss(input_batch).numpy()

            valid_loss = valid_loss/self.nvalid

            # Check early stopping criteria
            if valid_loss < best_valid_loss:
                
                print('Improved validation loss from:',best_valid_loss,' to:', valid_loss)                
                best_valid_loss = valid_loss
                
                if self.arch_type == 'baseline':
                    self.save_weights('./checkpoints/baseline_checkpoint')
                elif self.arch_type == 'dropout':
                    self.save_weights('./checkpoints/dropout_checkpoint')
                elif self.arch_type == 'mixture':
                    self.save_weights('./checkpoints/mixture_checkpoint')
                elif self.arch_type == 'ensemble':
                    self.save_weights('./checkpoints/ensemble_checkpoint')
                
                stop_iter = 0
            else:
                print('Validation loss (no improvement):',valid_loss)
                stop_iter = stop_iter + 1

            if stop_iter == patience:
                break
                
    # Load weights
    def restore_model(self):
        if self.arch_type == 'baseline':
            self.load_weights('./checkpoints/baseline_checkpoint')
        elif self.arch_type == 'dropout':
            self.load_weights('./checkpoints/dropout_checkpoint')
        elif self.arch_type == 'mixture':
            self.load_weights('./checkpoints/mixture_checkpoint')
        elif self.arch_type == 'ensemble':
            self.load_weights('./checkpoints/ensemble_checkpoint')

    # Do some testing
    def model_inference(self,mc_num=100):
        # Restore from checkpoint
        self.restore_model()

        if self.arch_type == 'baseline':

            predictions, latent = self.call(self.test_data)

            np.save('Predicted_baseline.npy',predictions.numpy())
        
        elif self.arch_type == 'dropout':
            
            prediction_list = []
            for i in range(mc_num):
                recon, _ = self.call(self.test_data)
                recon = recon.numpy()
                prediction_list.append(recon)

            prediction_list = np.asarray(prediction_list)
            np.save('MC_Dropout_predictions.npy',prediction_list)

        
        elif self.arch_type == 'mixture' or self.arch_type == 'ensemble':       

            mean, logvar, _ = self.call(self.test_data)
            mean = mean.numpy()
            logvar = logvar.numpy()

            if self.arch_type == 'ensemble':
                np.save('Deep_Ensembles_Mean.npy',mean)
                np.save('Deep_Ensembles_LV.npy',logvarsquare)
            else:
                np.save('Mixtures_Mean.npy',mean)
                np.save('Mixtures_LV.npy',logvarsquare)


def load_partition_data():
    # Load data
    data = np.load('./snapshot_matrix_pod.npy')[:4096].T

    # Scale the training data
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    # Transpose (rows are DOF, columns are snapshots)
    data = data.T

    swe_train_data = np.zeros(shape=(np.shape(data)[1],64,64,1)) # Channels last
    for i in range(np.shape(data)[1]):
        temp_1 = data[0:64*64,i].reshape(64,64)
        swe_train_data[i,:,:,0] = np.transpose(temp_1[:,:])

    # Randomize train
    idx =  np.arange(swe_train_data.shape[0])
    np.random.shuffle(idx)
    swe_train_data_randomized = swe_train_data[idx[:]].astype('float32')

    swe_valid_data = swe_train_data_randomized[720:]
    swe_train_data = swe_train_data_randomized[:720]
        
    data = np.load('./snapshot_matrix_test.npy')[:4096].T
    data = scaler.transform(data)
    # Transpose (rows are DOF, columns are snapshots)
    data = data.T

    swe_test_data = np.zeros(shape=(np.shape(data)[1],64,64,1)) # Channels last
    for i in range(np.shape(data)[1]):
        temp_1 = data[0:64*64,i].reshape(64,64)
        swe_test_data[i,:,:,0] = np.transpose(temp_1[:,:])

    swe_test_data = swe_test_data.astype('float32')

    data_tuple = (swe_train_data, swe_valid_data, swe_test_data)

    return data_tuple

if __name__ == '__main__':
    data_tuple = load_partition_data()

    eps_range = 0.01*(np.max(data_tuple[0])-np.min(data_tuple[0]))
    model = cnn_autoencoder(data_tuple,arch_type='mixture',eps_range=eps_range)
    # model.train_model()
    model.model_inference()
    
        
