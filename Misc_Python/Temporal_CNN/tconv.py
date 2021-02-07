# Temporal convolutional neural network
# Refer https://github.com/philipperemy/keras-tcn

import numpy as np
np.random.seed(10)
import tensorflow as tf
tf.set_random_seed(10)

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from tcn import TCN

batch_size, timesteps, input_dim = None, 20, 1


def get_x_y(size=1000):
    import numpy as np
    pos_indices = np.random.choice(size, size=int(size // 2), replace=False)
    x_train = np.zeros(shape=(size, timesteps, 1))
    y_train = np.zeros(shape=(size, 1))
    x_train[pos_indices, 0] = 1.0
    y_train[pos_indices, 0] = 1.0
    return x_train, y_train


i = Input(batch_shape=(batch_size, timesteps, input_dim))

o = TCN(return_sequences=False)(i)  # The TCN layers are here.
o = Dense(1)(o)

m = Model(inputs=[i], outputs=[o])
m.compile(optimizer='adam', loss='mse')

x, y = get_x_y()
m.fit(x, y, epochs=10, validation_split=0.2)