import numpy as np
import matplotlib.pyplot as plt

temp_cuda = np.genfromtxt('Temperature.txt')

dx = 2.0*np.pi/np.shape(temp_cuda)[0]
dy = 2.0*np.pi/np.shape(temp_cuda)[1]

x,y = np.meshgrid(np.arange(0,2.0*np.pi,dx), np.arange(0,2.0*np.pi,dy))

plt.figure()
plt.contourf(x,y,temp_cuda)
plt.colorbar()
plt.title('Final condition')
plt.show()