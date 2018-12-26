import numpy as np
import matplotlib.pyplot as plt



solution = np.genfromtxt('Results.txt')
nx = int(np.sqrt(np.shape(solution)[0]))
ny = nx

dx = 2.0*np.pi/float(nx)
dy = 2.0*np.pi/float(ny)

u = solution[:,0]
v = solution[:,1]
p = solution[:,2]

u = np.transpose(np.reshape(u,newshape=(nx,ny)))
v = np.transpose(np.reshape(v,newshape=(nx,ny)))
p = np.transpose(np.reshape(p,newshape=(nx,ny)))

print(np.shape(solution))

x,y = np.meshgrid(np.arange(0,2.0*np.pi,dx), np.arange(0,2.0*np.pi,dy))

fig, ax = plt.subplots(nrows=2,ncols=2)

ax[0,0].set_title('U')
a = ax[0,0].contourf(x,y,u)
pa = fig.colorbar(a,ax=ax[0,0])

ax[0,1].set_title('V')
b = ax[0,1].contourf(x,y,v)
pb = fig.colorbar(b,ax=ax[0,1])

ax[1,0].set_title('P')
c = ax[1,0].contourf(x,y,p)
pc = fig.colorbar(c,ax=ax[1,0])

ax[1,1].set_frame_on(False)
ax[1,1].axes.get_yaxis().set_visible(False)
ax[1,1].axes.get_xaxis().set_visible(False)
ax[1,1].text(0.2,0.7,r'Cavity flow solution')
ax[1,1].text(0.2,0.6,r'Second-order accurate')
ax[1,1].text(0.2,0.5,r'Staggered grid')


plt.show()