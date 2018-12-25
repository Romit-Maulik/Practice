import numpy as np
import matplotlib.pyplot as plt

nx_global = 512
ny_global = 512
num_procs_x = 4
num_procs_y = 4

nx = nx_global // num_procs_x
ny = nx

dx = 2.0*np.pi/nx_global
dy = 2.0*np.pi/ny_global

u_global = np.zeros(shape=(nx_global,ny_global),dtype='double')
x, y = np.meshgrid(np.arange(0, 2.0*np.pi, step=dx), np.arange(0, 2.0*np.pi, step=dy))

for xid in range(num_procs_x):
    for yid in range(num_procs_y):
        filename = str(xid) + r'_' + str(yid) + '.npy'
        u_local = np.load(filename)

        u_local = u_local[1:nx+1,1:ny+1]

        u_global[xid*nx:(xid+1)*nx, yid*ny:(yid+1)*ny] = u_local[:,:]


fig, ax = plt.subplots(nrows=1,ncols=1)

p1 = ax.contourf(x,y,u_global)
plt.colorbar(p1,ax=ax)#,format="%.2f"
plt.show()