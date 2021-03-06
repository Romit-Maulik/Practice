import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt

class online_svd_calculator(object):
    """
    docstring for online_svd_calculator:
    K : Number of modes to truncate
    ff : Forget factor
    """
    def __init__(self, K, ff):
        super(online_svd_calculator, self).__init__()
        self.K = K
        self.ff = ff

    def initialize(self,A):
        # Computing R-SVD of the initial matrix - step 1 section II
        q, r = np.linalg.qr(A)

        # Compute SVD of r - v is already transposed  - step 2 section II
        # https://stackoverflow.com/questions/24913232/using-numpy-np-linalg-svd-for-singular-value-decomposition
        ui, self.di, self.vit = np.linalg.svd(r) 

        # Get back U and truncate
        self.ui = np.matmul(q,ui)[:,:self.K]  #- step 3 section II
        self.di = self.di[:self.K]

    def incorporate_data(self,A):
        """
        A is the new data matrix
        """
        # Section III B 3(a):
        m_ap = self.ff*np.matmul(self.ui,np.diag(self.di))
        m_ap = np.concatenate((m_ap,A),axis=-1)
        udashi, ddashi = np.linalg.qr(m_ap)

        # Section III B 3(b):
        utildei, dtildei, vtildeti = np.linalg.svd(ddashi)

        # Section III B 3(c):
        max_idx = np.argsort(dtildei)[::-1][:self.K]
        self.di = dtildei[max_idx]
        utildei = utildei[:,max_idx]
        self.ui = np.matmul(udashi,utildei)

    def plot_modes(self):
        plt.figure()
        plt.plot(self.ui[:,0],label='Mode 0')
        plt.plot(self.ui[:,1],label='Mode 1')
        plt.plot(self.ui[:,2],label='Mode 2')
        plt.plot(self.ui[:,3],label='Mode 3')
        plt.legend()
        plt.show()

        
if __name__ == '__main__':
    test_class = online_svd_calculator(10,0.95)
    # Load data
    initial_data = np.load('Burgers_train_snapshots.npy')[:100,:,0].T
    new_data = np.load('Burgers_train_snapshots.npy')[100:120,:,0].T
    newer_data = np.load('Burgers_train_snapshots.npy')[120:150,:,0].T

    # Do a first modal decomposition and visualize
    test_class.initialize(initial_data)
    test_class.plot_modes()

    # Incorporate new data and visualize
    test_class.incorporate_data(new_data)
    test_class.plot_modes()

    # Incorporate newer data and visualize
    test_class.incorporate_data(new_data)
    test_class.plot_modes()