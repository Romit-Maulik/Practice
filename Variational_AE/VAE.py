import numpy as np
import tensorflow as tf

# Set seeds
np.random.seed(10)
tf.random.set_seed(10)

from tensorflow.keras.layers import Input, Dense, LSTM, Lambda, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D

from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import binary_crossentropy, mse

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt

lrate = 0.001
weights_filepath = 'best_weights_vae.h5'
mode = 'train' # train, test
num_latent = 2

def model_def():
    
    def coeff_determination(y_pred, y_true): #Order of function inputs is important here        
        SS_res =  K.sum(K.square( y_true-y_pred )) 
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )

    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # then z = z_mean + sqrt(var)*eps
    def sampling(args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
         Arguments
            args (tensor): mean and log of variance of Q(z|X)
         Returns
            z (tensor): sampled latent vector
        """

        epsilon_mean = 0.1
        epsilon_std = 1e-4

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim), mean=epsilon_mean, stddev=epsilon_std)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    ## Encoder
    encoder_inputs = Input(shape=(64,64,1),name='Field')
    # Encode   
    x = Conv2D(30,kernel_size=(3,3),activation='relu',padding='same')(encoder_inputs)
    enc_l2 = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

    x = Conv2D(25,kernel_size=(3,3),activation='relu',padding='same')(enc_l2)
    enc_l3 = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

    x = Conv2D(20,kernel_size=(3,3),activation='relu',padding='same')(enc_l3)
    enc_l4 = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

    x = Conv2D(15,kernel_size=(3,3),activation='relu',padding='same')(enc_l4)
    enc_l5 = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

    x = Conv2D(10,kernel_size=(3,3),activation=None,padding='same')(enc_l5)
    encoded = MaxPooling2D(pool_size=(2, 2),padding='same')(x)

    x = Flatten()(x)
    z_mean = Dense(num_latent, name='z_mean')(x)
    z_log_var = Dense(num_latent, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(num_latent,), name='z')([z_mean, z_log_var])
    # instantiate encoder model
    encoder = Model(encoder_inputs, z, name='encoder')

    # build decoder model
    latent_inputs = Input(shape=(num_latent,), name='z_sampling')
    x = Dense(8)(latent_inputs)
    x = Reshape((2, 2, 2))(x)
       
    x = Conv2D(2,kernel_size=(3,3),activation=None,padding='same')(x)
    dec_l1 = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(15,kernel_size=(3,3),activation='relu',padding='same')(dec_l1)
    dec_l2 = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(20,kernel_size=(3,3),activation='relu',padding='same')(dec_l2)
    dec_l3 = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(25,kernel_size=(3,3),activation='relu',padding='same')(dec_l3)
    dec_l4 = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(30,kernel_size=(3,3),activation='relu',padding='same')(dec_l4)
    dec_l5 = UpSampling2D(size=(2, 2))(x)

    decoded = Conv2D(1,kernel_size=(3,3),activation=None,padding='same')(dec_l5)
    decoder = Model(inputs=latent_inputs,outputs=decoded)
    decoder.summary()
    # instantiate VAE model
    ae_outputs = decoder(encoder(encoder_inputs))
    model = Model(inputs=encoder_inputs,outputs=ae_outputs,name='VAE')

    # Losses and optimization
    my_adam = optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    reconstruction_loss = mse(K.flatten(encoder_inputs), K.flatten(ae_outputs))
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    model.add_loss(vae_loss)
    model.compile(optimizer=my_adam,metrics=[coeff_determination])
    model.summary()

    return model, decoder, encoder


# Grab data
swe_train = np.load('./snapshot_matrix_pod.npy').T[:,:4096]
swe_valid = np.load('./snapshot_matrix_test.npy').T[:,:4096]

preproc = Pipeline([('stdscaler', StandardScaler())])
swe_train = preproc.fit_transform(swe_train)
swe_valid = preproc.transform(swe_valid)
swe_train = swe_train.reshape(900,64,64,1)
swe_valid = swe_valid.reshape(100,64,64,1)

# Shuffle - to preserve the order of the initial dataset
swe_train_data = np.copy(swe_train)
swe_valid_data = np.copy(swe_valid)

np.random.shuffle(swe_train_data)
np.random.shuffle(swe_valid_data)

if __name__ == '__main__':

    model,decoder,encoder = model_def()

    # CNN training stuff
    num_epochs = 5000
    batch_size = 4

    # fit network
    if mode == 'train':
        checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=True)
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
        callbacks_list = [checkpoint,earlystopping]
        train_history = model.fit(x=swe_train_data, y=swe_train_data, epochs=num_epochs, batch_size=batch_size, callbacks=callbacks_list, validation_split=0.1)

        # model.load_weights(weights_filepath)

        # Encode the training data to generate time-series information
        encoded_t = K.eval(encoder(swe_train[:,:,:,:].astype('float32')))[0].numpy()
        encoded_v = K.eval(encoder(swe_valid[:,:,:,:].astype('float32')))[0].numpy()

        encoded_t = encoded_t.reshape(900,num_latent)
        encoded_v = encoded_v.reshape(100,num_latent)

        plt.figure()
        plt.plot(encoded_t[0:10,0],label='Dimension 1')
        plt.plot(encoded_t[0:10,1],label='Dimension 2')
        plt.legend()
        plt.show()

        np.save('VAE_Coefficient_Training_Data.npy',encoded_t)
        np.save('VAE_Coefficient_Testing_Data.npy',encoded_v)

    if mode == 'test':
        # Visualize fields
        model.load_weights(weights_filepath)

        # Metric calculation
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

        predicted = preproc.inverse_transform(model.predict(swe_valid).reshape(100,4096))
        true = preproc.inverse_transform(swe_valid.reshape(100,4096))
        
        print('R2 score:',r2_score(true,predicted))
        print('MSE score:',mean_squared_error(true,predicted))
        print('MAE score:',mean_absolute_error(true,predicted))

        
        for time in range(0,10):
            recoded = model.predict(swe_valid[time:time+1,:,:,:])
            true = preproc.inverse_transform(swe_valid[time:time+1,:,:,:].reshape(1,4096)).reshape(64,64)
            recoded = preproc.inverse_transform(recoded.reshape(1,4096)).reshape(64,64)

            np.save('True_'+str(time)+'.npy',true)
            np.save('Rec_'+str(time)+'.npy',recoded)

            fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(6,6))
            cs1 = ax[0].imshow(true,label='input')
            cs2 = ax[1].imshow(recoded,label='decoded')

            for i in range(2):
                ax[i].set_xlabel('x')
                ax[i].set_ylabel('y')
                    
            fig.colorbar(cs1,ax=ax[0],fraction=0.046, pad=0.04)
            fig.colorbar(cs2,ax=ax[1],fraction=0.046, pad=0.04)
            ax[0].set_title(r'True $q_1$')
            ax[1].set_title(r'Reconstructed $q_1$')
            plt.subplots_adjust(wspace=0.5,hspace=-0.3)
            plt.tight_layout()
            plt.show()

