import numpy as np
import matplotlib.pyplot as plt


#Post process MPI output from C
num_procs = 16
nx_global = 256
ny_global = 256


proc_dim = int(np.sqrt(num_procs))
lx_global = 2.0*np.pi
ly_global = 2.0*np.pi
dx = lx_global/nx_global
dy = ly_global/ny_global


nx_local = nx_global//proc_dim
ny_local = ny_global//proc_dim
lx_local = 2.0*np.pi/proc_dim
ly_local = 2.0*np.pi/proc_dim


id_range = np.arange(num_procs,dtype='int')

x_id = np.remainder(id_range,proc_dim) 
y_id = id_range//proc_dim

x,y = np.meshgrid(dx*np.arange(1,nx_global+1), dy*np.arange(1,ny_global+1))
u = np.copy(x)

for i in range(num_procs):
	sol = np.genfromtxt(str(i))
	xloc = x_id[i]
	yloc = y_id[i]

	u[xloc*nx_local:(xloc+1)*nx_local,yloc*ny_local:(yloc+1)*ny_local] = sol[0:nx_local,0:ny_local]

plt.figure()
plt.contourf(x,y,u)
plt.colorbar()
plt.title('Final solution')
plt.show()