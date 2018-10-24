import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint

def create_keras_model(features, targets):

    #Import parameters from dict
    n_inputs = np.shape(features)[1]

    #Import training data from dict
    training_inputs = features
    training_outputs = targets

    # initialization of turbulence models basis model
    model = Sequential()
    # Layers start
    input_layer = Input(shape=(n_inputs,))

    # Hidden layers
    x = Dense(50, activation='tanh', use_bias=True)(input_layer)
    x = Dense(50, activation='tanh', use_bias=True)(x)

    op_val = Dense(np.shape(targets)[1], activation='linear', use_bias=True)(x)

    custom_model = Model(inputs=input_layer, outputs=op_val)

    filepath = "best_model.hd5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    custom_model.compile(optimizer='adam', loss='mean_squared_error')
    history_callback = custom_model.fit(training_inputs, training_outputs, epochs=500, batch_size=32, verbose=1,callbacks=callbacks_list)

    loss_history = history_callback.history["loss"]

    loss_history = np.array(loss_history)
    np.savetxt("loss_history.txt", loss_history, delimiter=",")

    return custom_model

def make_md_training_data():
    n_samples = 1000    #Number of training data samples
    n_inputs = 2        #Number of input parameters
    n_outputs = 3       #Number of output parameters

    #Define arrays for storing these
    input_data = np.zeros(shape=(n_samples, n_inputs), dtype='double')
    output_data = np.zeros(shape=(n_samples, n_outputs), dtype='double')

    #Populate arrays
    np.random.seed(1)
    for i in range(n_samples):
        x = np.random.uniform(low=0.0, high=2.0 * np.pi)
        y = np.random.uniform(low=0.0, high=2.0 * np.pi)

        input_data[i, 0] = x
        input_data[i, 1] = y

        output_data[i, 0] = np.sin(x)*np.sin(y)
        output_data[i, 1] = np.sin(x)+np.cos(y)
        output_data[i, 2] = np.sin(-x-y)

    return input_data, output_data

def plot_data(inputs,outputs):

    fig = plt.figure()
    ax = fig.add_subplot(311, projection='3d')
    ax.plot_trisurf(inputs[:,0],inputs[:,1],outputs[:,0],cmap=cm.jet, linewidth=0.2)
    ax.set_title('Function 1')
    ax.grid(False)
    ax.axis('off')

    ax = fig.add_subplot(312, projection='3d')
    ax.plot_trisurf(inputs[:,0],inputs[:,1],outputs[:,1],cmap=cm.jet, linewidth=0.2)
    ax.set_title('Function 2')
    ax.grid(False)
    ax.axis('off')


    ax = fig.add_subplot(313, projection='3d')
    ax.plot_trisurf(inputs[:,0],inputs[:,1],outputs[:,2],cmap=cm.jet, linewidth=0.2)
    ax.set_title('Function 3')
    ax.grid(False)
    ax.axis('off')

    plt.legend()
    plt.show()


    plt.figure()
    f1_true = outputs[:, 0].flatten()
    f2_true = outputs[:, 1].flatten()
    f3_true = outputs[:, 2].flatten()
    plt.hist(f1_true, bins=16, label=r'Function 1', histtype='step')  # arguments are passed to np.histogram
    plt.hist(f2_true, bins=16, label=r'Function 2', histtype='step')  # arguments are passed to np.histogram
    plt.hist(f3_true, bins=16, label=r'Function 3', histtype='step')  # arguments are passed to np.histogram
    plt.legend()
    plt.show()

def check_md_model_performance(model):
    n_samples = 5000  # Number of training data samples
    n_inputs = 2  # Number of input parameters

    # Define arrays for storing these
    input_data = np.zeros(shape=(n_samples, n_inputs), dtype='double')

    # Populate arrays
    np.random.seed(2)
    for i in range(n_samples):
        x = np.random.uniform(low=0.0, high=2.0 * np.pi)
        y = np.random.uniform(low=0.0, high=2.0 * np.pi)

        input_data[i, 0] = x
        input_data[i, 1] = y

    pred = model.predict(input_data)

    return input_data, pred


features, targets = make_md_training_data()
plot_data(features,targets)
model = create_keras_model(features,targets)
test_inputs, pred = check_md_model_performance(model)
plot_data(test_inputs, pred)