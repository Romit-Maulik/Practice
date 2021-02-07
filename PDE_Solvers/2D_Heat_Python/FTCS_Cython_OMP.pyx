#cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
from cython.parallel import prange, parallel

cpdef loop_ftcs_cython_omp(int nx, int ny, double dx, double dy, double dt, double ft, double alpha, double[:,:] u):

	cdef int nt = int(ft/dt)
	cdef int i, j, t
	cdef int nxc = nx
	cdef int nyc = ny

	cdef double const_mult = alpha*dt/(dx*dx)
	cdef double [:,:] ump = np.zeros((nx+2,ny+2))
	cdef double [:,:] utemp = np.zeros((nx+2,ny+2))

	for i in range(0,nx+2):
			for j in range(0,ny+2):
				utemp[i,j] = u[i,j]
				ump[i,j] = u[i,j]

	with nogil:
		t = 0
		while t<nt:
			t = t+1

			# Array copy
			for i in prange(0,nxc+2):
				for j in prange(0,nyc+2):
					utemp[i,j] = ump[i,j]

			
			#FTCS timestep
			for i in prange(1,nxc+1):
				for j in prange(1,nyc+1):
					ump[i,j] = utemp[i,j] + const_mult*(-4.0*utemp[i,j]+(utemp[i-1,j]+utemp[i+1,j]+utemp[i,j-1]+utemp[i,j+1]))

			#Periodic boundary condition update
			for j in prange(1,nyc+1):
				ump[0,j] = ump[nxc,j]
				ump[nxc+1,j] = ump[1,j]

			for i in prange(0,nxc+2):
				ump[i,0] = ump[i,nyc]
				ump[i,nyc+1] = ump[i,1]

	for i in range(0,nx+2):
		for j in range(0,ny+2):
			u[i,j] = ump[i,j]
					
	return u