import numpy as np
import matplotlib.pyplot as plt
import time

def init_domain():
	#Solution domain
	global nx,ny,dx,dy,dt,alpha,ft
	nx = 512
	ny = 512
	
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

	return u,x,y


def plot_field(x,y,u,title):
	#levels = np.linspace(-1, 1, 20)
	fig, ax = plt.subplots(nrows=1,ncols=1)
	ax.set_title(title)
	p1 = ax.contourf(x,y,u)
	plt.colorbar(p1,ax=ax)#,format="%.2f"
	plt.show()

def vectorize_ftcs_1(u):

	global dx,dy,dt,alpha,ft
	nt = int(ft/dt)
	const_mult = alpha*dt/(dx*dx)
	
	#Finite difference coefficients
	i,j = np.meshgrid(np.arange(1,nx+1),np.arange(1,ny+1))
	ip1,jp1 = np.meshgrid(np.arange(2,nx+2),np.arange(2,ny+2))
	im1,jm1 = np.meshgrid(np.arange(0,nx),np.arange(0,ny))

	#Indices for BC update - x direction
	i1,j1 = np.meshgrid(np.arange(0,1),np.arange(1,ny+1))
	i2,j2 = np.meshgrid(np.arange(nx,nx+1),np.arange(1,ny+1))

	i3,j3 = np.meshgrid(np.arange(1,2),np.arange(1,ny+1))
	i4,j4 = np.meshgrid(np.arange(nx+1,nx+2),np.arange(1,ny+1))

	i5,j5 = np.meshgrid(np.arange(0,nx+2),np.arange(0,1))
	i6,j6 = np.meshgrid(np.arange(0,nx+2),np.arange(ny,ny+1))

	i7,j7 = np.meshgrid(np.arange(0,nx+2),np.arange(1,2))
	i8,j8 = np.meshgrid(np.arange(0,nx+2),np.arange(ny+1,ny+2))

	t = 0
	while t<nt:
		t = t+1
		utemp = np.copy(u)		
		u[i,j] = utemp[i,j] + const_mult*(-4.0*utemp[i,j]+(utemp[im1,j]+utemp[ip1,j]+utemp[i,jm1]+utemp[i,jp1]))

		#Update BCS
		#Need to update BCs - x direction
		u[i1,j1] = u[i2,j2]
		u[i4,j4] = u[i3,j3]

		#Need to update BCs - y direction
		u[i5,j5] = u[i6,j6]
		u[i8,j8] = u[i7,j7]

		del(utemp)

	return u


def vectorize_ftcs_2(u):

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

def loop_ftcs(u):

	global dx,dy,dt,alpha,ft
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

if __name__ == '__main__':
	
	global nx,ny,dx,dy,dt,alpha,ft
	uv, x, y = init_domain()
	plot_field(x,y,uv,'Initial Condition')

	t1 = time.time()
	uv = vectorize_ftcs_2(uv)
	print('CPU time taken for vectorized FTCS: ',time.time()-t1)
	plot_field(x,y,uv,'Final solution - vectorized FTCS')

	ul, x, y = init_domain()
	plot_field(x,y,ul,'Initial Condition')
	t1 = time.time()
	ul = loop_ftcs(ul)
	print('CPU time taken for for loop based FTCS: ',time.time()-t1)
	plot_field(x,y,ul,'Final solution - for loop based FTCS')

	plot_field(x,y,np.abs(ul-uv),'L1-norm difference (for loop and vectorized)')


