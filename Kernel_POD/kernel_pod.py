import numpy as np
np.random.seed(10)
from data_splitter import collect_snapshots
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt

num_components = 4

# http://fourier.eng.hmc.edu/e161/lectures/kernelPCA/node4.html

def centerK(K):
    ''' Returns centered K matrix '''
    M = K.shape[0]
    ot_ = np.ones(shape=(1,M))
    o_ = np.ones(shape=(M,1))
    o_ot = np.matmul(o_,ot_)

    Kcentered = K - 1.0/M*np.matmul(K,o_ot) - 1.0/M*np.matmul(o_ot,K)
    third_term = np.matmul(np.matmul(ot_,K),o_)
    third_term = 1.0/(M**2)*third_term[0,0]*o_ot

    Kcentered = Kcentered + third_term
    
    return Kcentered

def get_components(test_data,train_data,evecs):
    # Finding principal components in feature space    
    # Find K matrix (first for train data to check)
    kernelVals = rbf_kernel(test_data, train_data, gamma=0.1)

    BetaK = np.zeros(shape=(test_data.shape[0],num_components))
    for i in range(test_data.shape[0]):
        for k in range(num_components):
            BetaK[i,k] = np.sum(evecs[k]*kernelVals[i])

    return kernelVals, BetaK

if __name__ == '__main__':
    total_data, total_data_mean = collect_snapshots()
    total_data = total_data.T # Need to transpose for rbf kernel

    num_snapshots = np.shape(total_data)[0]
    num_dof = np.shape(total_data)[1]

    randomized = np.arange(num_snapshots)
    np.random.shuffle(randomized)

    train_data = total_data[randomized[:300]]
    test_data = total_data[randomized[300:]]

    K = centerK(rbf_kernel(train_data,gamma=0.1))

    # Solve eigenvalue problem for Kernel matrix
    evals, evecs = np.linalg.eig(K)
    evals = evals/np.shape(K)[0]

    # Drop negative Evals
    for i, l in enumerate(evals):
        if l < 10**(-8):
            evals = evals[:i]
            evecs = evecs[:i]
            break

    evals = evals[:num_components].astype('double') # This will flag a warning for cast - ignore it
    evecs = evecs[:num_components].astype('double') # This will flag a warning for cast - ignore it

    print('Train data kernel matrix shape:',K.shape)

    _, BetaK_all = get_components(total_data,train_data,evecs)

    print('K-PCA shape for all data:',BetaK_all.shape)

    plt.figure()
    plt.plot(BetaK_all[:,0],label='K-PCA dimension 1')
    plt.plot(BetaK_all[:,1],label='K-PCA dimension 2')
    plt.plot(BetaK_all[:,2],label='K-PCA dimension 3')
    plt.plot(BetaK_all[:,3],label='K-PCA dimension 4')
    plt.legend()
    plt.title('K-PCA evolution over time')
    plt.show()

    # Learning pre image
    # Objective function would be || K x - A BetaK.T ||_2 with A as decision variable (for simplest approach)

    # Learning pre-images from training data alone
    kernelVals, BetaK_train = get_components(train_data,train_data,evecs)
    Kx = np.matmul(kernelVals,train_data)

    # Optimizing for A in num_components x num_dof
    def residual(A):
        A = A.reshape(num_components,num_dof)
        return np.sum((Kx - np.matmul(BetaK_train,A))**2)

    callback_array = np.zeros(shape=(1,num_components*num_dof))
    def callbackF(Xi):
        global callback_array
        sol_array = np.copy(Xi)
        callback_array = np.concatenate((callback_array,sol_array.reshape(1,-1)),axis=0)

    from scipy.optimize import minimize
    solution = minimize(residual,np.zeros(shape=(num_components*num_dof)),method='L-BFGS-B',
                            tol=1e-8,options={'disp': True, 'maxfun':10000000, 'eps': 1.4901161193847656e-8}, 
                            callback=callbackF)

    Aopt = solution.x
    print(Aopt.reshape(num_components,num_dof))

    np.save('Optimized_Preimage.npy',Aopt.reshape(num_components,num_dof))