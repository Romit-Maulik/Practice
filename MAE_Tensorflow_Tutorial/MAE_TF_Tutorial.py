import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import os
import time


def make_training_data():
    n_samples = 500    #Number of training data samples
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

    return n_samples, n_inputs, n_outputs, input_data, output_data


def make_testing_data():
    n_samples = 5000    #Number of training data samples
    n_inputs = 2        #Number of input parameters

    #Define arrays for storing these
    input_data = np.zeros(shape=(n_samples, n_inputs), dtype='double')

    #Populate arrays
    np.random.seed(2)
    for i in range(n_samples):
        x = np.random.uniform(low=0.0, high=2.0 * np.pi)
        y = np.random.uniform(low=0.0, high=2.0 * np.pi)

        input_data[i, 0] = x
        input_data[i, 1] = y

    return input_data


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



def define_weights_biases(n_inputs,n_outputs):

    l1_nodes = 50
    l2_nodes = 50

    #Truncated normal distribution - values greater than 2 standard deviations are dropped and repicked

    weights = {
        'l1': tf.Variable(tf.truncated_normal(shape=[n_inputs, l1_nodes], seed=1, mean=0.0, stddev=0.1)),
        'l2': tf.Variable(tf.truncated_normal([l1_nodes, l2_nodes], seed=1, mean=0.0, stddev=0.1)),
        'l3': tf.Variable(tf.truncated_normal([l2_nodes, n_outputs], seed=1, mean=0.0, stddev=0.1))
    }

    biases = {
        'l1': tf.Variable(tf.truncated_normal([l1_nodes], seed=1, mean=0.0, stddev=0.1)),
        'l2': tf.Variable(tf.truncated_normal([l2_nodes], seed=1, mean=0.0, stddev=0.1)),
        'l3': tf.Variable(tf.truncated_normal([n_outputs], seed=1, mean=0.0, stddev=0.1))
    }

    return weights, biases


def feed_forward(x,weights,biases):
    # Inputs of weights and biases as dictionaries
    l1 = tf.add(tf.matmul(x, weights['l1']), biases['l1'])
    l1 = tf.nn.sigmoid(l1)

    l2 = tf.add(tf.matmul(l1, weights['l2']), biases['l2'])
    l2 = tf.nn.sigmoid(l2)

    prediction = tf.add(tf.matmul(l2, weights['l3']), biases['l3'])

    return prediction

def our_cost(y,y_true):
    #Simple MSE error
    return tf.reduce_mean(tf.squared_difference(y,y_true))


def placeholders(n_inputs,n_outputs):
    #Inputs to DNN
    x_true = tf.placeholder(tf.float32,shape=[None, n_inputs])
    #Outputs
    y_true = tf.placeholder(tf.float32,shape=[None, n_outputs])

    return x_true, y_true


def train_my_neural_network():
    # Load data and sizes
    n_samples, n_inputs, n_outputs, input_data, output_data = make_training_data()

    #Check plots
    plot_data(input_data,output_data)

    t0 = time.time()
    #Start network code
    batch_size = 200
    x_true, y_true = placeholders(n_inputs, n_outputs)

    # Initialize weights and biases
    weights, biases = define_weights_biases(n_inputs, n_outputs)
    prediction = feed_forward(x_true,weights,biases)
    cost = our_cost(prediction,y_true)


    with tf.Session() as sess:
        #prediction = feed_forward(x_true,weights,biases)
        #cost = our_cost(prediction,y_true)
        train_step = tf.train.AdamOptimizer().minimize(cost)

        sess.run(tf.global_variables_initializer())

        hm_epochs = 10000
        epoch = 0

        # For visualization
        epoch_loss_array = np.zeros((hm_epochs, 3), dtype='double')

        while epoch < hm_epochs:
            epoch_loss = 0
            epoch_loss_val = 0


            for _ in range(int(n_samples / batch_size)):
                idx1 = np.random.randint(n_samples, size=batch_size)
                epoch_ip, epoch_op = input_data[idx1, :], output_data[idx1, :]

                #ycheck = np.ones(shape=(batch_size,n_outputs),dtype=float)

                # c = sess.run(func_check,feed_dict={x: epoch_ip, y_: ycheck})
                # print(c)
                # print(np.shape(c))
                # os.system('pause')

                train_step.run(session=sess,
                               feed_dict={x_true: epoch_ip, y_true: epoch_op})
                c = sess.run(cost, feed_dict={x_true: epoch_ip, y_true: epoch_op})

                epoch_loss = epoch_loss + c

                idx1 = np.random.randint(n_samples, size=batch_size)
                epoch_ip_val, epoch_op_val = input_data[idx1, :], output_data[idx1, :]

                c_val = sess.run(cost, feed_dict={x_true: epoch_ip_val, y_true: epoch_op_val})

                epoch_loss_val = epoch_loss_val + c_val

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            print('Validation loss:', epoch_loss_val)

            epoch_loss_array[epoch, 0] = epoch
            epoch_loss_array[epoch, 1] = epoch_loss
            epoch_loss_array[epoch, 2] = epoch_loss_val

            epoch = epoch + 1

        t1 = time.time()
        print('Time = ',t1-t0)

        # Plotting training performance
        plt.figure()
        plt.title('Performance')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.semilogx(epoch_loss_array[:, 0], epoch_loss_array[:, 1], label='Total training loss')
        plt.semilogx(epoch_loss_array[:, 0], epoch_loss_array[:, 2], label='Total validation loss')
        plt.legend()
        plt.show()

        # Training done - save weights
        w1_val = sess.run(weights['l1'])
        w2_val = sess.run(weights['l2'])
        w3_val = sess.run(weights['l3'])

        b1_val = sess.run(biases['l1'])
        b2_val = sess.run(biases['l2'])
        b3_val = sess.run(biases['l3'])

        return w1_val, w2_val, w3_val, b1_val, b2_val, b3_val


def network_prediction(x, w1, w2, w3, b1, b2, b3):
    # Inputs of weights and biases as dictionaries
    l1 = np.add(np.matmul(x, w1), b1)
    l1 = sigmoid(l1)

    l2 = np.add(np.matmul(l1, w2), b2)
    l2 = sigmoid(l2)

    f = np.add(np.matmul(l2, w3), b3)

    return f

def sigmoid(x):
  return 1/(1+np.exp(-x))

if __name__ == "__main__":
    w1, w2, w3, b1, b2, b3 = train_my_neural_network()

    testing_data = make_testing_data()
    testing_outputs = network_prediction(testing_data, w1, w2, w3, b1, b2, b3)

    plot_data(testing_data,testing_outputs)
