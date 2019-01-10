import numpy as np
import matplotlib.pyplot as plt

temp_init = np.genfromtxt('Temperature.txt')

dx = 2.0*np.pi/np.shape(temp_init)[0]
dy = 2.0*np.pi/np.shape(temp_init)[1]

x,y = np.meshgrid(np.arange(0,2.0*np.pi,dx), np.arange(0,2.0*np.pi,dy))

plt.figure()
plt.contourf(temp_init)
plt.colorbar()
plt.title('Final condition')
plt.show()
