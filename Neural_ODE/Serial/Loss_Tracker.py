import numpy as np
import matplotlib.pyplot as plt

mse_train = np.load('Train_Loss.npy')
mse_val = np.load('Val_Loss.npy')

plt.figure()
plt.semilogy(mse_train,label='Training')
plt.semilogy(mse_val,label='Validation')
plt.title('Error convergence - NODE')
plt.legend(fontsize=12)
plt.show()