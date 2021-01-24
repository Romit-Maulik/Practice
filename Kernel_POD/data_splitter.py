import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
import tensorflow as tf

Rnum = 2000.0
x = np.linspace(0.0,1.0,num=256)
dx = 1.0/np.shape(x)[0]

tsteps = np.linspace(0.0,2.0,num=800)
dt = 2.0/np.shape(tsteps)[0]

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# This is the Burgers problem definition
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
def exact_solution(t):
    t0 = np.exp(Rnum/8.0)

    return (x/(t+1))/(1.0+np.sqrt((t+1)/t0)*np.exp(Rnum*(x*x)/(4.0*t+4)))

def collect_snapshots():
    snapshot_matrix_total = np.zeros(shape=(np.shape(x)[0],np.shape(tsteps)[0]))

    trange = np.arange(np.shape(tsteps)[0])
    for t in trange:
        snapshot_matrix_total[:,t] = exact_solution(tsteps[t])[:]

    snapshot_matrix_mean = np.mean(snapshot_matrix_total,axis=1)
    snapshot_matrix = (snapshot_matrix_total.transpose()-snapshot_matrix_mean).transpose()

    return snapshot_matrix, snapshot_matrix_mean, snapshot_matrix_total

# Method of snapshots to accelerate
def method_of_snapshots(Y): #Mean removed
    '''
    Y - Snapshot matrix - shape: NxS
    '''
    new_mat = np.matmul(np.transpose(Y),Y)
    w, v = np.linalg.eig(new_mat)

    # Bases
    phi = np.matmul(Y,np.real(v))
    trange = np.arange(np.shape(Y)[1])
    phi[:,trange] = -phi[:,trange]/np.sqrt(np.abs(w)[:])

    return phi# POD modes

if __name__ == '__main__':
    
    # Collect data
    total_data, total_data_mean, _ = collect_snapshots()
    num_snapshots = total_data.shape[1]
    num_dof = total_data.shape[0]

    randomized = np.arange(num_snapshots)
    np.random.shuffle(randomized)

    train_data = total_data[:,randomized[:600]]
    test_data = total_data[:,randomized[600:]]

    # Generate serial POD with MOS
    modes = method_of_snapshots(train_data)
    # Check that this guy is close to one
    print('Inner product of similar modes summed:',np.sum(np.matmul(modes[:,1:2].T,modes[:,1:2])))
    # Check that this guy is close to zero
    print('Inner product of dissimilar modes summed:',np.sum(np.matmul(modes[:,1:2].T,modes[:,2:3])))

    # Find coefficient evolution of all data
    num_components = 4
    coeff_evolution = np.matmul(total_data.T,modes[:,:num_components])

    plt.figure()
    plt.plot(coeff_evolution[:,0],label='Dimension 1')
    plt.plot(coeff_evolution[:,1],label='Dimension 2')
    plt.plot(coeff_evolution[:,2],label='Dimension 3')
    plt.plot(coeff_evolution[:,3],label='Dimension 4')
    plt.legend()
    plt.title('Regular POD coefficient evolutions')
    plt.show()

    # Learning nonlinear function approximator from POD to reconstruction
    train_coeffs = np.matmul(train_data.T,modes[:,:num_components])
    test_coeffs = np.matmul(test_data.T,modes[:,:num_components])

    # Define NN model
    pod_inputs = tf.keras.Input(shape=(train_coeffs.shape[-1],))
    x = tf.keras.layers.Dense(30, activation="tanh")(pod_inputs)
    x = tf.keras.layers.Dense(30, activation="tanh")(x)
    outputs = tf.keras.layers.Dense(train_data.T.shape[-1])(x)

    model = tf.keras.Model(inputs=pod_inputs, outputs=outputs, name="inverse_image_model")

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        )   

    history = model.fit(train_coeffs, train_data.T, batch_size=128, epochs=10000, validation_split=0.1)
    
    # Try testing
    test_reconstruction = model.predict(test_coeffs)

    # Plot the reconstruction
    plt.figure()
    plt.plot(test_reconstruction[10,:],label="Predicted")
    plt.plot(test_data.T[10,:],label="True")
    plt.legend()
    plt.show()
