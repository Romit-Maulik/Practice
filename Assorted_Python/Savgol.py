import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

#Read data
file_data = np.loadtxt('Sav_Gol_Data.csv',dtype='double',delimiter=',')
print(np.shape(file_data))

plt.figure()
plt.plot(file_data[:,0],file_data[:,1])
plt.show()

# data_new = savgol_filter(file_data[:,2],131,polyorder=2)
#
# plt.figure()
# plt.plot(file_data[:,0],data_new[:])
# plt.show()
#
# np.savetxt('Sav_Gol_Filtered.csv',data_new,delimiter=',')
