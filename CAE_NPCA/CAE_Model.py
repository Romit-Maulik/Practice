import numpy as np
np.random.seed(10)
import tensorflow as tf
tf.random.set_seed(10)
tf.keras.backend.set_floatx('float64')
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.regularizers import l1

from CAE_Layers import Encoder_Block, Decoder_Block, Latent_Block
from sklearn.metrics import r2_score

def coeff_determination(y_pred, y_true): #Order of function inputs is important here        
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

class CAE_Model(Model):
    def __init__(self,data,num_latent,npca=False):
        super(CAE_Model, self).__init__()

        # Set up data for CAE
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        # Reshape for scaling
        data = data.reshape(1000,-1)
        self.num_latent = num_latent

        self.preproc = Pipeline([('stdscaler', StandardScaler())])

        self.ntrain = 700
        self.nvalid = 200
        self.ntest = 100

        self.swe_train = self.preproc.fit_transform(data[:self.ntrain])
        self.swe_valid = self.preproc.transform(data[self.ntrain:self.ntrain+self.nvalid])
        self.swe_test = self.preproc.transform(data[self.ntrain+self.nvalid:])

        self.swe_train = self.swe_train.reshape(self.ntrain,64,64,3)
        self.swe_valid = self.swe_valid.reshape(self.nvalid,64,64,3)
        self.swe_test = self.swe_test.reshape(self.ntest,64,64,3)

        # Shuffle - to preserve the order of the initial dataset
        self.swe_train_shuffled = np.copy(self.swe_train)
        self.swe_valid_shuffled = np.copy(self.swe_valid)

        np.random.shuffle(self.swe_train_shuffled)
        np.random.shuffle(self.swe_valid_shuffled)

        # Define architecture
        self.b1 = Encoder_Block()
        self.b2 = Latent_Block(self.num_latent)
        self.b3 = Decoder_Block()
        self.train_op = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Hierarchical PCA
        self.npca = npca

    # Running the model
    def call(self,X):
        if self.npca:
            h1 = self.b1(X)
            h2 = self.b2(h1)

            out_list = []
            for i in range(1,self.num_latent):
                h2_temp = h2.numpy()
                h2_temp[:,i:] = 0.0
                out_list.append(self.b3(tf.Variable(h2_temp)))

            out_list.append(self.b3(h2))
            return out_list
        
        else:
            h1 = self.b1(X)
            h2 = self.b2(h1)
            out = self.b3(h2)
        
            return out
    
    # Regular MSE
    def get_loss(self,X,Y):
        if self.npca:
            op_list = self.call(X)
            loss_val = tf.reduce_mean(tf.math.square(op_list[0]-Y))
            for i in range(1,self.num_latent):
                loss_val = loss_val + tf.reduce_mean(tf.math.square(op_list[i]-Y))
            return loss_val
        
        else:
            op=self.call(X)
            return tf.reduce_mean(tf.math.square(op-Y))

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
        patience = 10
        best_valid_loss = 999999.0 # Some large number 

        self.num_batches = 50
        self.train_batch_size = int(self.ntrain/self.num_batches)
        self.valid_batch_size = int((self.nvalid)/self.num_batches)
        
        for i in range(100):
            # Training loss
            print('Training iteration:',i)
            
            for batch in range(self.num_batches):
                input_batch = self.swe_train[batch*self.train_batch_size:(batch+1)*self.train_batch_size]
                output_batch = self.swe_train[batch*self.train_batch_size:(batch+1)*self.train_batch_size]
                self.network_learn(input_batch,output_batch)

            # Validation loss
            valid_loss = 0.0
            valid_r2 = 0.0

            for batch in range(self.num_batches):
                input_batch = self.swe_valid[batch*self.valid_batch_size:(batch+1)*self.valid_batch_size]
                output_batch = self.swe_valid[batch*self.valid_batch_size:(batch+1)*self.valid_batch_size]

                valid_loss = valid_loss + self.get_loss(input_batch,output_batch).numpy()

                if self.npca:
                    predictions = self.call(self.swe_valid)[-1].numpy()                    
                else:
                    predictions = self.call(self.swe_valid).numpy()
                valid_r2 = valid_r2 + r2_score(predictions.reshape(self.valid_batch_size,-1),self.swe_valid.reshape(self.valid_batch_size,-1))

            valid_r2 = valid_r2/(batch+1)


            # Check early stopping criteria
            if valid_loss < best_valid_loss:
                
                print('Improved validation loss from:',best_valid_loss,' to:', valid_loss)
                print('Validation R2:',valid_r2)
                
                best_valid_loss = valid_loss

                if self.npca:
                    self.save_weights('./npca_checkpoints/my_checkpoint')
                else:
                    self.save_weights('./cae_checkpoints/my_checkpoint')
                
                stop_iter = 0
            else:
                print('Validation loss (no improvement):',valid_loss)
                print('Validation R2:',valid_r2)
                stop_iter = stop_iter + 1

            if stop_iter == patience:
                break
                
        # Check accuracy on test
        if self.npca:
            predictions = self.call(self.swe_test)[-1].numpy()
        else:
            predictions = self.call(self.swe_test).numpy()
        print('Test loss:',self.get_loss(self.swe_test,self.swe_test).numpy())
        r2 = r2_score(predictions.reshape(self.valid_batch_size,-1),self.swe_test.reshape(self.valid_batch_size,-1))
        print('Test R2:',r2)
        r2_iter = 0

    # Load weights
    def restore_model(self):
        if self.npca:
            self.load_weights('./npca_checkpoints/my_checkpoint') # Load pretrained model
        else:
            self.load_weights('./cae_checkpoints/my_checkpoint') # Load pretrained model

    # Do some testing
    def model_inference(self):
        # Restore from checkpoint
        self.restore_model()

        # Reconstruct some test images
        if self.npca:
            for i in range(10):
                predicted_list = self.call(self.swe_test[i:i+1])

                # Rescale
                for j in range(self.num_latent):
                    predicted_list[j] = predicted_list[j].numpy()
                    predicted_list[j] = self.preproc.inverse_transform(predicted_list[j].reshape(1,-1)).reshape(1,64,64,3)
                
                true = self.preproc.inverse_transform(self.swe_test[i:i+1].reshape(1,-1)).reshape(1,64,64,3)

                self.plot_npca_comparison(true,predicted_list)
        else:
            for i in range(10):
                predicted = self.call(self.swe_test[i:i+1]).numpy()

                # Rescale
                predicted = self.preproc.inverse_transform(predicted.reshape(1,-1)).reshape(1,64,64,3)
                true = self.preproc.inverse_transform(self.swe_test[i:i+1].reshape(1,-1)).reshape(1,64,64,3)

                self.plot_standard_comparison(true,predicted)


    def plot_standard_comparison(self,true,predicted):
        
        fig, ax = plt.subplots(nrows=3,ncols=2,figsize=(6,8))
        cs1 = ax[0,0].imshow(true[0,:,:,0],label='input')
        cs2 = ax[0,1].imshow(predicted[0,:,:,0],label='decoded')

        fig.colorbar(cs1,ax=ax[0,0],fraction=0.046, pad=0.04)
        fig.colorbar(cs2,ax=ax[0,1],fraction=0.046, pad=0.04)

        ax[0,0].set_title(r'True $q_1$')
        ax[0,1].set_title(r'Reconstructed $q_1$')

        cs1 = ax[1,0].imshow(true[0,:,:,1],label='input')
        cs2 = ax[1,1].imshow(predicted[0,:,:,1],label='decoded')

        fig.colorbar(cs1,ax=ax[1,0],fraction=0.046, pad=0.04)
        fig.colorbar(cs2,ax=ax[1,1],fraction=0.046, pad=0.04)

        ax[1,0].set_title(r'True $q_2$')
        ax[1,1].set_title(r'Reconstructed $q_2$')

        cs1 = ax[2,0].imshow(true[0,:,:,2],label='input')
        cs2 = ax[2,1].imshow(predicted[0,:,:,2],label='decoded')

        fig.colorbar(cs1,ax=ax[2,0],fraction=0.046, pad=0.04)
        fig.colorbar(cs2,ax=ax[2,1],fraction=0.046, pad=0.04)

        ax[2,0].set_title(r'True $q_3$')
        ax[2,1].set_title(r'Reconstructed $q_3$')

        for i in range(2):
            for j in range(2):
                ax[i,j].set_xlabel('x')
                ax[i,j].set_ylabel('y')
                
        
        plt.subplots_adjust(wspace=0.5,hspace=-0.3)
        plt.tight_layout()
        plt.show()


    def plot_npca_comparison(self,true,predicted_list):
        
        for i in range(self.num_latent):
            predicted = predicted_list[i]

            fig, ax = plt.subplots(nrows=3,ncols=2,figsize=(6,8))
            cs1 = ax[0,0].imshow(true[0,:,:,0],label='input')
            cs2 = ax[0,1].imshow(predicted[0,:,:,0],label='decoded')

            fig.colorbar(cs1,ax=ax[0,0],fraction=0.046, pad=0.04)
            fig.colorbar(cs2,ax=ax[0,1],fraction=0.046, pad=0.04)

            ax[0,0].set_title(r'True $q_1$')
            ax[0,1].set_title(r'Reconstructed $q_1$')

            cs1 = ax[1,0].imshow(true[0,:,:,1],label='input')
            cs2 = ax[1,1].imshow(predicted[0,:,:,1],label='decoded')

            fig.colorbar(cs1,ax=ax[1,0],fraction=0.046, pad=0.04)
            fig.colorbar(cs2,ax=ax[1,1],fraction=0.046, pad=0.04)

            ax[1,0].set_title(r'True $q_2$')
            ax[1,1].set_title(r'Reconstructed $q_2$')

            cs1 = ax[2,0].imshow(true[0,:,:,2],label='input')
            cs2 = ax[2,1].imshow(predicted[0,:,:,2],label='decoded')

            fig.colorbar(cs1,ax=ax[2,0],fraction=0.046, pad=0.04)
            fig.colorbar(cs2,ax=ax[2,1],fraction=0.046, pad=0.04)

            ax[2,0].set_title(r'True $q_3$')
            ax[2,1].set_title(r'Reconstructed $q_3$')

            for i in range(2):
                for j in range(2):
                    ax[i,j].set_xlabel('x')
                    ax[i,j].set_ylabel('y')
                    
            
            plt.subplots_adjust(wspace=0.5,hspace=-0.3)
            plt.tight_layout()
        
        plt.show()


if __name__ == '__main__':
    print('CAE Model defined in this module')