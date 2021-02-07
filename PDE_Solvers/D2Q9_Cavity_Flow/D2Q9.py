#Refer to Bao and Meskas tutorial on D2Q9 LBM
import numpy as np
import matplotlib.pyplot as plt

# Global variables
nx = 128
ny = 128

dx = 1.0
dy = 1.0

nt = 1000
dt = 1.0

cval = dx/dt

lid_vel = 0.5
Re_n = 100.0

nu_val = float(nx)*lid_vel/Re_n
tau_val = 0.5*(6.0*nu_val/(cval*dx)+1.0)

print('The tau value is: ',tau_val)
print('The nu is: ', nu_val)


# Weights for equilibrium distribution 
w = np.asarray([4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0])

def velocity_decompose(u,v):
	u_decompose = np.zeros(shape=(nx,ny,9),dtype='double')
	v_decompose = np.zeros(shape=(nx,ny,9),dtype='double')

	u_decompose[:,:,0] = 0.0
	u_decompose[:,:,1] = u[:,:]
	u_decompose[:,:,2] = 0.0
	u_decompose[:,:,3] = -u[:,:]
	u_decompose[:,:,4] = 0.0
	u_decompose[:,:,5] = u[:,:]
	u_decompose[:,:,6] = -u[:,:]
	u_decompose[:,:,7] = -u[:,:]
	u_decompose[:,:,8] = u[:,:]


	v_decompose[:,:,0] = 0.0
	v_decompose[:,:,1] = 0.0
	v_decompose[:,:,2] = v[:,:]
	v_decompose[:,:,3] = 0.0
	v_decompose[:,:,4] = -v[:,:]
	v_decompose[:,:,5] = v[:,:]
	v_decompose[:,:,6] = v[:,:]
	v_decompose[:,:,7] = -v[:,:]
	v_decompose[:,:,8] = -v[:,:]

	decompose = u_decompose + v_decompose

	return decompose

def calculate_s_i(u,v):
	# Calculate decomposed velocities
	u_dec = velocity_decompose(u,v)

	# Constant norm of velocity
	const_val = np.zeros(shape=(nx,ny),dtype='double')
	const_val[:,:] = -1.5/(cval*cval)*(u[:,:]*u[:,:] + v[:,:]*v[:,:])

	# Calculate s values for equilibrium calculation
	s_val = np.zeros(shape=(nx,ny,9),dtype='double')
	# e_0
	s_val[:,:,0] = w[0]*(const_val[:,:])
	# e_1
	s_val[:,:,1] = w[1]*(3.0/cval*(u_dec[:,:,1]) + 9.0/(2.0*cval*cval)*(u_dec[:,:,1]**2)+const_val[:,:])
	# e_2
	s_val[:,:,2] = w[2]*(3.0/cval*(u_dec[:,:,2]) + 9.0/(2.0*cval*cval)*(u_dec[:,:,2]**2)+const_val[:,:])
	# e_3
	s_val[:,:,3] = w[3]*(3.0/cval*(u_dec[:,:,3]) + 9.0/(2.0*cval*cval)*(u_dec[:,:,3]**2)+const_val[:,:])
	# e_4
	s_val[:,:,4] = w[4]*(3.0/cval*(u_dec[:,:,4]) + 9.0/(2.0*cval*cval)*(u_dec[:,:,4]**2)+const_val[:,:])
	# e_5
	s_val[:,:,5] = w[5]*(3.0/cval*(u_dec[:,:,5]) + 9.0/(2.0*cval*cval)*(u_dec[:,:,5]**2)+const_val[:,:])
	# e_6
	s_val[:,:,6] = w[6]*(3.0/cval*(u_dec[:,:,6]) + 9.0/(2.0*cval*cval)*(u_dec[:,:,6]**2)+const_val[:,:])
	# e_7
	s_val[:,:,7] = w[7]*(3.0/cval*(u_dec[:,:,7]) + 9.0/(2.0*cval*cval)*(u_dec[:,:,7]**2)+const_val[:,:])
	# e_8
	s_val[:,:,8] = w[8]*(3.0/cval*(u_dec[:,:,8]) + 9.0/(2.0*cval*cval)*(u_dec[:,:,8]**2)+const_val[:,:])

	return s_val

def calculate_eq(u,v,rho):
	s_val = calculate_s_i(u,v)
	f_eq = np.zeros(shape=(nx,ny,9),dtype='double')
	range_val = np.arange(0,9)
	f_eq[:,:,range_val] = w[None,None,range_val]*rho[:,:,None] + rho[:,:,None]*s_val[:,:,range_val]

	return f_eq

def initialize_problem():

	# Initialize equilibrium distribution
	u = np.zeros(shape=(nx,ny),dtype='double')# U velocity
	v = np.zeros(shape=(nx,ny),dtype='double')# V velocity
	rho = np.ones(shape=(nx,ny),dtype='double')
	#Assign lid velocity
	u[1:nx-1,ny-1] = lid_vel

	# Distribution calculations
	f_eq = calculate_eq(u,v,rho)	
	f = np.copy(f_eq)

	return f, f_eq, rho, u, v

def calculate_density(f):
	rho = np.sum(f,axis=2)
	return rho

def calculate_velocity(f,rho):
	u = f[:,:,1] - f[:,:,3] + (f[:,:,5]+f[:,:,8]) - (f[:,:,6]+f[:,:,7])
	v = f[:,:,2] - f[:,:,4] + (f[:,:,5]+f[:,:,6]) - (f[:,:,7]+f[:,:,8])

	u[:,:] = 1.0/rho[:,:]*(cval*u[:,:])
	v[:,:] = 1.0/rho[:,:]*(cval*v[:,:])

	return u, v

def streaming_boundaries(f,f_temp,f_eq,rho,u,v):
	#----------------------------------------------------------------------------#
	#----------------------------------------------------------------------------#
	# Left wall updated except top left corner - where velocity boundaries are used
	f[0,0:ny-1,1] = f_temp[0,0:ny-1,3] #Validated bounce back

	f[0,1:ny-1,2] = f_temp[0,0:ny-2,2]
	f[0,0,2] = f_temp[0,0,4]#Bounce back lower left corner - Validated

	f[0,0:ny-1,3] = f_temp[1,0:ny-1,3]#Validated

	f[0,0:ny-1,4] = f_temp[0,1:ny,4]# Validated
	

	f[0,0:ny-1,5] = f_temp[0,0:ny-1,7]# Validated - bounce back

	f[0,1:ny-1,6] = f_temp[1,0:ny-2,6]
	f[0,0,6] = f_temp[0,0,8]# Validated


	f[0,0:ny-1,7] = f_temp[1,1:ny,7]# Validated
	

	f[0,0:ny-1,8] = f_temp[0,0:ny-1,6]	#Validated bounce back


	#----------------------------------------------------------------------------#
	#----------------------------------------------------------------------------#
	# Right wall updated except top right corner - where velocity boundaries are used
	f[nx-1,0:ny-1,1] = f_temp[nx-2,0:ny-1,1] # Validated

	f[nx-1,1:ny-1,2] = f_temp[nx-1,0:ny-2,2]
	f[nx-1,0,2] = f_temp[nx-1,0,4]#Bounce back lower right corner - Validated

	f[nx-1,0:ny-1,3] = f_temp[nx-1,0:ny-1,1]#Validated bounce back

	f[nx-1,0:ny-1,4] = f_temp[nx-1,1:ny,4]# Validated
	
	f[nx-1,1:ny-1,5] = f_temp[nx-2,0:ny-2,5]# Validated
	f[nx-1,0,5] = f_temp[nx-1,0,7]# Validated

	f[nx-1,0:ny-1,6] = f_temp[nx-1,0:ny-1,8] #Validated bounce back
	
	f[nx-1,0:ny-1,7] = f_temp[nx-1,0:ny-1,5]# Validated bounce back
	
	f[nx-1,0:ny-1,8] = f_temp[nx-2,1:ny,8]	#Validated

	#----------------------------------------------------------------------------#
	#----------------------------------------------------------------------------#
	# Lower wall except bottom corners (taken care of previously)
	f[1:nx-1,0,1] = f_temp[0:nx-2,0,1] # Validated

	f[1:nx-1,0,2] = f_temp[1:nx-1,0,4] # Validated bounce back

	f[1:nx-1,0,3] = f_temp[2:nx,0,3] # Validated

	f[1:nx-1,0,4] = f_temp[1:nx-1,1,4] # Validated 

	f[1:nx-1,0,5] = f_temp[1:nx-1,0,7] # Validated bounce back

	f[1:nx-1,0,6] = f_temp[1:nx-1,0,8] # Validated bounce back

	f[1:nx-1,0,7] = f_temp[2:nx,1,7] # Validated

	f[1:nx-1,0,8] = f_temp[0:nx-2,1,8] # Validated


	# ----------------------------------------------------------------------------------
	# ----------------------------------------------------------------------------------

	# Lid - velocity BC - needs to be fixed

	f[1:nx,ny-1,1] = f_temp[0:nx-1,ny-1,1]
	f[0,ny-1,1] = f_temp[0,ny-1,3] # Validated - bounce back

	f[0:nx,ny-1,2] = f_temp[0:nx,ny-2,2] # Validated

	f[0:nx-1,ny-1,3] = f_temp[1:nx,ny-1,3] # Validated
	f[nx-1,ny-1,3] = f_temp[nx-1,ny-1,1] # Validated bounce back

	f[0:nx,ny-1,4] = f_temp[0:nx,ny-1,2]

	f[1:nx,ny-1,5] = f_temp[0:nx-1,ny-2,5]
	f[0,ny-1,5] = f_temp[0,ny-1,7] # Validated - corner bounce back only

	f[0:nx-1,ny-1,6] = f_temp[1:nx,ny-2,6]
	f[nx-1,ny-1,6] = f_temp[nx-1,ny-1,8] # Validated - corner bounce back only

	rhon = f_temp[1:nx-1,ny-1,0] + f_temp[1:nx-1,ny-1,1] + f_temp[1:nx-1,ny-1,3] + 2.0*(f_temp[1:nx-1,ny-1,2]+f_temp[1:nx-1,ny-1,6]+f_temp[1:nx-1,ny-1,5])

	f[1:nx-1,ny-1,7] = f_temp[1:nx-1,ny-1,5] + 0.5*(f_temp[1:nx-1,ny-1,1]-f_temp[1:nx-1,ny-1,3])-0.5*rhon*lid_vel
	f[0,ny-1,7] = f_temp[0,ny-1,5]
	f[nx-1,ny-1,7] = f_temp[nx-1,ny-1,5]

	f[1:nx-1,ny-1,8] = f_temp[1:nx-1,ny-1,6] + 0.5*(f_temp[1:nx-1,ny-1,3]-f_temp[1:nx-1,ny-1,1])+0.5*rhon*lid_vel
	f[0,ny-1,8] = f_temp[0,ny-1,6]
	f[nx-1,ny-1,8] = f_temp[nx-1,ny-1,6]

	return f

def streaming(f,f_eq,rho,u,v):
	# Temporary copy of f
	f_temp = np.copy(f)
	# Boundary conditions update
	f = streaming_boundaries(f,f_temp,f_eq,rho,u,v)
	
	#Define ranges for within the domain - f0
	nx_internal, ny_internal = np.meshgrid(np.arange(1,nx-1), np.arange(1,ny-1))
	f[:,:,0] = f_temp[:,:,0]

	#Define ranges for within the domain - f1
	nx_neighbor, ny_neighbor = np.meshgrid(np.arange(0,nx-2), np.arange(1,ny-1))
	f[nx_internal,ny_internal,1] = f_temp[nx_neighbor,ny_neighbor,1]

	#Define ranges for within the domain - f2
	nx_neighbor, ny_neighbor = np.meshgrid(np.arange(1,nx-1), np.arange(0,ny-2))
	f[nx_internal,ny_internal,2] = f_temp[nx_neighbor,ny_neighbor,2]


	#Define ranges for within the domain - f3
	nx_neighbor, ny_neighbor = np.meshgrid(np.arange(2,nx), np.arange(1,ny-1))
	f[nx_internal,ny_internal,3] = f_temp[nx_neighbor,ny_neighbor,3]

	#Define ranges for within the domain - f4
	nx_neighbor, ny_neighbor = np.meshgrid(np.arange(1,nx-1), np.arange(2,ny))
	f[nx_internal,ny_internal,4] = f_temp[nx_neighbor,ny_neighbor,4]

	#Define ranges for within the domain - f5
	nx_neighbor, ny_neighbor = np.meshgrid(np.arange(0,nx-2), np.arange(0,ny-2))
	f[nx_internal,ny_internal,5] = f_temp[nx_neighbor,ny_neighbor,5]

	#Define ranges for within the domain - f6
	nx_neighbor, ny_neighbor = np.meshgrid(np.arange(2,nx), np.arange(0,ny-2))
	f[nx_internal,ny_internal,6] = f_temp[nx_neighbor,ny_neighbor,6]

	#Define ranges for within the domain - f7
	nx_neighbor, ny_neighbor = np.meshgrid(np.arange(2,nx), np.arange(2,ny))
	f[nx_internal,ny_internal,7] = f_temp[nx_neighbor,ny_neighbor,7]

	#Define ranges for within the domain - f8
	nx_neighbor, ny_neighbor = np.meshgrid(np.arange(0,nx-2), np.arange(2,ny))
	f[nx_internal,ny_internal,8] = f_temp[nx_neighbor,ny_neighbor,8]

	del f_temp

	return f

def collision_step(f,f_eq):
	f[:,:,:] = f[:,:,:] - 1.0/tau_val*(f[:,:,:]-f_eq[:,:,:])

def time_loop(f,f_eq,rho,u,v):

	t = 0

	print('Macroscopic density sum: ',np.sum(rho))
	while t<nt:
		t = t+1

		# Calculate new equilibrium distribution
		f_eq = calculate_eq(u,v,rho)

		# Collision step
		collision_step(f,f_eq)

		# Streaming
		streaming(f,f_eq,rho,u,v)

		# Calculate physical quantities from updated distribution
		rho = calculate_density(f)
		u, v = calculate_velocity(f,rho)

		# Setting BCs
		u[1:nx-1,ny-1] = lid_vel
		v[1:nx-1,ny-1] = 0.0

		print('Macroscopic density sum: ',np.sum(rho), 'Time step: ',t)



	plt.figure()
	plt.title('U')
	p1 = plt.contourf(np.transpose(u))
	plt.colorbar(p1)
	plt.show()

	plt.figure()
	plt.title('V')
	p1 = plt.contourf(np.transpose(v))
	plt.colorbar(p1)
	plt.show()

	plt.figure()
	plt.title('Macroscopic density')
	p1 = plt.contourf(np.transpose(rho))
	plt.colorbar(p1)
	plt.show()
	

f, f_eq, rho, u, v = initialize_problem()
time_loop(f,f_eq,rho,u,v)


