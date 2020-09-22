import numpy as np
np.random.seed(10)
import tensorflow as tf
tf.random.set_seed(10)
tf.keras.backend.set_floatx('float64')


from tensorflow.keras.layers import Input, Dense, LSTM, Lambda, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D
from tensorflow.keras.layers import Layer

class Encoder_Block(Layer):
    def __init__(self):
        super(Encoder_Block, self).__init__()
        print('Initialized encoder block')

    def build(self,input_shape):
        # Encode   
        self.l1_1 = Conv2D(30,kernel_size=(3,3),activation='relu',padding='same')
        self.l1_2 = MaxPooling2D(pool_size=(2, 2),padding='same')

        self.l2_1 = Conv2D(25,kernel_size=(3,3),activation='relu',padding='same')
        self.l2_2 = MaxPooling2D(pool_size=(2, 2),padding='same')

        self.l3_1 = Conv2D(20,kernel_size=(3,3),activation='relu',padding='same')
        self.l3_2 = MaxPooling2D(pool_size=(2, 2),padding='same')

        self.l4_1 = Conv2D(15,kernel_size=(3,3),activation='relu',padding='same')
        self.l4_2 = MaxPooling2D(pool_size=(2, 2),padding='same')

        self.l5_1 = Conv2D(10,kernel_size=(3,3),activation=None,padding='same')
        self.l5_2 = MaxPooling2D(pool_size=(2, 2),padding='same')
        self.lf = Flatten()

    def call(self, inputs):
        x = self.l1_1(inputs)
        x = self.l1_2(x)

        x = self.l2_1(x)
        x = self.l2_2(x)

        x = self.l3_1(x)
        x = self.l3_2(x)

        x = self.l4_1(x)
        x = self.l4_2(x)

        x = self.l5_1(x)
        x = self.l5_2(x)

        y = self.lf(x)
        return y

class Latent_Block(Layer):
    def __init__(self,num_latent):
        super(Latent_Block, self).__init__()
        self.num_latent = num_latent
        print('Initialized latent block')

    def build(self,input_shape):
        self.fll = Flatten()
        self.fcl = Dense(self.num_latent,activation=None)

    def call(self,inputs):
        x = self.fll(inputs)
        y = self.fcl(x)

        return y

class Decoder_Block(Layer):
    def __init__(self):# Input dim is hard set to 2 [x1, x2], Output dim is 2 [y1, y2]
        super(Decoder_Block, self).__init__()
        print('Initialized Decoder block')
    def build(self,input_shape):
        # Decode
        self.l1_1 = Dense(8,activation=None)
        self.l1_2 = Reshape(target_shape=(2,2,2))

        self.l2_1 = Conv2D(10,kernel_size=(3,3),activation=None,padding='same')
        self.l2_2 = UpSampling2D(size=(2, 2))

        self.l3_1 = Conv2D(15,kernel_size=(3,3),activation='relu',padding='same')
        self.l3_2 = UpSampling2D(size=(2, 2))

        self.l4_1 = Conv2D(20,kernel_size=(3,3),activation='relu',padding='same')
        self.l4_2 = UpSampling2D(size=(2, 2))

        self.l5_1 = Conv2D(25,kernel_size=(3,3),activation='relu',padding='same')
        self.l5_2 = UpSampling2D(size=(2, 2))

        self.l6_1 = Conv2D(30,kernel_size=(3,3),activation='relu',padding='same')
        self.l6_2 = UpSampling2D(size=(2, 2))

        self.l7_1 = Conv2D(3,kernel_size=(3,3),activation=None,padding='same')

    def call(self, inputs):
        x = self.l1_1(inputs)
        x = self.l1_2(x)

        x = self.l2_1(x)
        x = self.l2_2(x)

        x = self.l3_1(x)
        x = self.l3_2(x)

        x = self.l4_1(x)
        x = self.l4_2(x)

        x = self.l5_1(x)
        x = self.l5_2(x)

        x = self.l6_1(x)
        x = self.l6_2(x)

        y = self.l7_1(x)

        return y

if __name__ == '__main__':
    print('CAE Layers defined in this module')

    # Test
    data_input = np.random.uniform(size=(1,64,64,3))
    b1 = Encoder_Block()
    b2 = Latent_Block(8)
    b3 = Decoder_Block()

    x = b1(data_input)
    x = b2(x)
    x = b3(x)

    print(x.numpy().shape)