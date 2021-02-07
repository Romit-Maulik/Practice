# Parallel training visualization
import numpy as np
import matplotlib.pyplot as plt

total_procs = 4
plt.figure()
for i in range(total_procs):
	filename_train_loss = 'loss_history_'+str(i)+'.txt'
	arr = np.loadtxt(filename_train_loss)
	plt.plot(arr,label='Processor '+str(i))
plt.legend()
plt.show()