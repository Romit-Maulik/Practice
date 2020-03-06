import numpy as np

#pythran export loop_ftcs_pythran(int, int, float, float, float, float, float, float[][])
def loop_ftcs_pythran(nx,ny,dx,dy,dt,ft,alpha,u):

	nt = int(ft/dt)
	const_mult = alpha*dt/(dx*dx)
	utemp = np.zeros((nx+2,ny+2))

	t = 0
	while t<nt:
		t = t+1

		for i in range(0,nx+2):
			for j in range(0,ny+2):
				utemp[i,j] = u[i,j]
		
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
				
	return u