import matplotlib.pyplot as plt
import numpy as np
import time
from FTCS_Pythran import loop_ftcs_pythran
from FTCS_Cython import loop_ftcs_cython
from FTCS_Cython_OMP import loop_ftcs_cython_omp

nx = 128
ny = 128

#Cell sizes
dx = 2.0*np.pi/float(nx)
dy = 2.0*np.pi/float(ny)

#Indices
index_range = np.arange(0,nx+2)
x, y = np.meshgrid(dx*index_range-dx,dy*index_range-dy)

#diffusion parameter
alpha = 0.8
#Stable timestep
dt = 0.8*dx*dx/(4.0*alpha)

#Initialize
u = np.sin(x+y)

#length of solution
ft = 1.0

def plot_field(x,y,u,title):
	#levels = np.linspace(-1, 1, 20)
	fig, ax = plt.subplots(nrows=1,ncols=1)
	ax.set_title(title)
	p1 = ax.contourf(x,y,u)
	plt.colorbar(p1,ax=ax)#,format="%.2f"
	plt.show()

def loop_ftcs(u):

	nt = int(ft/dt)
	const_mult = alpha*dt/(dx*dx)

	t = 0
	while t<nt:
		t = t+1
		utemp = np.copy(u)		
		
		#FTCS timestep
		for i in range(1,nx+1):
			for j in range(1,ny+1):
				u[i,j] = utemp[i,j] + const_mult*(-4.0*utemp[i,j]+(utemp[i-1,j]+utemp[i+1,j]+utemp[i,j-1]+utemp[i,j+1]))

		#Periodic boundary condition update
		for j in range(1,ny+1):
			u[0,j] = u[nx,j]
			u[nx+1,j] = u[1,j]

		for i in range(0,nx+2):
			u[i,0] = u[i,ny]
			u[i,ny+1] = u[i,1]
				
		del(utemp)

	return u

def vectorized_ftcs(u):

	global dx,dy,dt,alpha,ft
	nt = int(ft/dt)
	const_mult = alpha*dt/(dx*dx)

	t = 0
	while t<nt:
		t = t+1
		utemp = np.copy(u)
		u[1:nx+1,1:ny+1] = utemp[1:nx+1,1:ny+1] + const_mult*(-4.0*utemp[1:nx+1,1:ny+1]+(utemp[0:nx,1:ny+1]+
							utemp[2:nx+2,1:ny+1]+utemp[1:nx+1,0:ny]+utemp[1:nx+1,2:ny+2]))

		#Update BCS
		#Need to update BCs - x direction
		u[0:1,1:ny+1] = u[nx:nx+1,1:ny+1]
		u[nx+1:nx+2,1:ny+1] = u[1:2,1:ny+1]

		#Need to update BCs - y direction
		u[0:nx+2,0:1] = u[0:nx+2,ny:ny+1]
		u[0:nx+2,ny+1:ny+2] = u[0:nx+2,1:2]

		del(utemp)

	return u


if __name__ == '__main__':
	
	compile_option = int(input('Enter 0 for native, 1 for vectorized, 2 for Cython build, 3 for Pythran build, 4 for Cython_OMP: '))

	if compile_option == 0:
		print('This will take some time for large values of nx. Be warned.')
		plot_field(x,y,u,'Initial Condition')
		t1 = time.time()
		u = loop_ftcs(u)
		print('CPU time taken for for loop based FTCS: ',time.time()-t1)
		plot_field(x,y,u,'Final solution')
	elif compile_option == 1:
		plot_field(x,y,u,'Initial Condition')
		t1 = time.time()
		u = vectorized_ftcs(u)
		print('CPU time taken for vectorized FTCS: ',time.time()-t1)
		plot_field(x,y,u,'Final solution')
	elif compile_option == 2:
		plot_field(x,y,u,'Initial Condition')
		t1 = time.time()
		u = loop_ftcs_cython(nx,ny,dx,dy,dt,ft,alpha,u)
		print('CPU time taken for Cython build: ',time.time()-t1)
		plot_field(x,y,u,'Final solution')
	elif compile_option == 3:
		plot_field(x,y,u,'Initial Condition')
		t1 = time.time()
		u = loop_ftcs_pythran(nx,ny,dx,dy,dt,ft,alpha,u)
		print('CPU time taken for for Pythran build: ',time.time()-t1)
		plot_field(x,y,u,'Final solution')
	elif compile_option == 4:
		plot_field(x,y,u,'Initial Condition')
		t1 = time.time()
		u = loop_ftcs_cython_omp(nx,ny,dx,dy,dt,ft,alpha,u)
		print('CPU time taken for Cython OMP build: ',time.time()-t1)
		plot_field(x,y,u,'Final solution')
