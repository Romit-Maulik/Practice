#mpiexec -n 4 python3 MPI_Heat.py
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time

# Global variables
nx_global = 512
ny_global = 512
dx = 2.0*np.pi/nx_global
dy = 2.0*np.pi/ny_global
ft = 1.0
alpha = 0.8
dt = 0.8*dx*dx/(4.0*alpha)


#MPI tags
send_up_tag = 0
send_down_tag = 1
send_left_tag = 2
send_right_tag = 3

def plot_field(x,y,u,title):
	fig, ax = plt.subplots(nrows=1,ncols=1)
	ax.set_title(title)
	p1 = ax.contourf(x,y,u)
	plt.colorbar(p1,ax=ax)#,format="%.2f"
	plt.show()

def write_field(u,rank,nprocs):
	num_procs_x = int(np.sqrt(nprocs))
	num_procs_y = int(np.sqrt(nprocs))

	# Rank in each dimension
	xid = np.remainder(rank, num_procs_x)
	yid = rank // num_procs_y

	filename = str(xid)+r'_'+str(yid)+'.npy'
	np.save(filename,u)

def initiate_local_domain(rank,nprocs):
	num_procs_x = int(np.sqrt(nprocs))
	num_procs_y = int(np.sqrt(nprocs))

	# num_points_x = nx_global // num_procs_x
	# num_points_y = ny_global // num_procs_y

	# Rank in each dimension
	xid = np.remainder(rank, num_procs_x)
	yid = rank // num_procs_y

	# Delimit boundaries of the local domain
	xstart = xid * 2.0 * np.pi / num_procs_x
	ystart = yid * 2.0 * np.pi / num_procs_y

	index_range = np.arange(0,nx_global//num_procs_x+2)
	x, y = np.meshgrid(xstart + index_range*dx - dx, ystart + index_range*dy - dy)

	# Populate array with initial condition
	local_domain = np.sin(x+y)
	local_domain = local_domain.astype('float64')

	return x,y,local_domain

def exchange_boundary_information(u_local,comm,rank,nprocs):
	num_procs_x = int(np.sqrt(nprocs))
	num_procs_y = int(np.sqrt(nprocs))

	num_points_x = nx_global // num_procs_x
	num_points_y = ny_global // num_procs_y

	# Rank in each dimension
	xid = np.remainder(rank, num_procs_x)
	yid = rank // num_procs_y

	# Buffer array
	buffer_array_send = np.empty(num_points_x + 2, dtype='double')
	buffer_array_recv = np.empty(num_points_x + 2, dtype='double')

	# Send all data - horizontal directions
	# Send right commands for periodic bcs
	if xid != num_procs_x - 1:
		# buffer_array_send[:] = u_local[num_points_x,:]
		# comm.Send([buffer_array_send, MPI.DOUBLE], dest=rank+1, tag=send_right_tag)
		comm.Send([u_local[num_points_x,:], MPI.DOUBLE], dest=rank + 1, tag=send_right_tag)

	# Send left commands for periodic bcs
	if xid != 0:
		# buffer_array_send[:] = u_local[1,:]
		# comm.Send([buffer_array_send, MPI.DOUBLE], dest=rank-1, tag=send_left_tag)
		comm.Send([u_local[1,:], MPI.DOUBLE], dest=rank - 1, tag=send_left_tag)

	# Receive all data - horizontal directions
	# Receive from right
	if xid != num_procs_x - 1:
		# comm.Recv([buffer_array_recv, MPI.DOUBLE], source=rank+1, tag=send_left_tag)
		# u_local[num_points_x + 1, :] = buffer_array_recv[:]
		comm.Recv([u_local[num_points_x + 1, :], MPI.DOUBLE], source=rank + 1, tag=send_left_tag)

	#Receive from left
	if xid != 0:
		# comm.Recv([buffer_array_recv, MPI.DOUBLE], source=rank-1, tag=send_right_tag)
		# u_local[0,:] = buffer_array_recv[:]
		comm.Recv([u_local[0, :], MPI.DOUBLE], source=rank - 1, tag=send_right_tag)

	# Send all data vertical directions
	# Send top commands for periodic bcs
	if yid != num_procs_y - 1:
		buffer_array_send[:] = u_local[:, num_points_y]
		comm.Send([buffer_array_send, MPI.DOUBLE], dest=rank + num_procs_x, tag=send_up_tag)

	# Send bottom commands for periodic bcs
	if yid != 0:
		buffer_array_send[:] = u_local[:, 1]
		comm.Send([buffer_array_send, MPI.DOUBLE], dest=rank - num_procs_x, tag=send_down_tag)

	# Receive all data vertical directions
	#Receive from top
	if yid != num_procs_y - 1:
		comm.Recv([buffer_array_recv, MPI.DOUBLE],source=rank+num_procs_x,tag=send_down_tag)
		u_local[:,num_points_y+1] = buffer_array_recv[:]

	# Receive from bottom
	if yid != 0:
		comm.Recv([buffer_array_recv, MPI.DOUBLE],source=rank-num_procs_x,tag=send_up_tag)
		u_local[:,0] = buffer_array_recv[:]

	# Periodic boundaries update
	# Horizontal directions
	if xid == 0: #Sending left
		target_id = rank + num_procs_x - 1
		# buffer_array_send[:] = u_local[1,:]
		# comm.Send([buffer_array_send,MPI.DOUBLE],dest=target_id,tag=send_left_tag)
		comm.Send([u_local[1,:], MPI.DOUBLE], dest=target_id, tag=send_left_tag)

	if xid == num_procs_x-1:#Sending right
		target_id = rank - num_procs_x + 1
		# buffer_array_send[:] = u_local[num_points_x, :]
		# comm.Send([buffer_array_send, MPI.DOUBLE], dest=target_id, tag=send_right_tag)
		comm.Send([u_local[num_points_x, :], MPI.DOUBLE], dest=target_id, tag=send_right_tag)

	if xid == num_procs_x - 1:#Receiving to the right
		source_id = rank - num_procs_x + 1
		# comm.Recv([buffer_array_recv,MPI.DOUBLE],source=source_id,tag=send_left_tag)
		# u_local[num_points_x+1,:] = buffer_array_recv[:]
		comm.Recv([u_local[num_points_x + 1, :], MPI.DOUBLE], source=source_id, tag=send_left_tag)

	if xid == 0:#Receiving to the left
		source_id = rank + num_procs_x - 1
		# comm.Recv([buffer_array_recv, MPI.DOUBLE], source=source_id, tag=send_right_tag)
		# u_local[0,:] = buffer_array_recv[:]
		comm.Recv([u_local[0, :], MPI.DOUBLE], source=source_id, tag=send_right_tag)

	# Vertical direction
	if yid == 0:#Sending bottom
		target_id = rank + (num_procs_y-1)*num_procs_y
		buffer_array_send[:] = u_local[:,1]
		comm.Send([buffer_array_send,MPI.DOUBLE],dest=target_id,tag=send_down_tag)

	if yid == num_procs_y - 1:#Sending top
		target_id = rank - (num_procs_y-1)*num_procs_y
		buffer_array_send[:] = u_local[:, num_points_y]
		comm.Send([buffer_array_send,MPI.DOUBLE],dest=target_id,tag=send_up_tag)

	if yid == num_procs_y - 1:
		source_id = rank - (num_procs_y-1)*num_procs_y
		comm.Recv([buffer_array_recv,MPI.DOUBLE],source=source_id,tag=send_down_tag)
		u_local[:,num_points_y+1] = buffer_array_recv[:]

	if yid == 0:
		source_id = rank + (num_procs_y-1)*num_procs_y
		comm.Recv([buffer_array_recv,MPI.DOUBLE],source=source_id,tag=send_up_tag)
		u_local[:, 0] = buffer_array_recv[:]

	del buffer_array_recv, buffer_array_send
	#Completed all message passing - blocking communications

def ftcs_update(u,utemp,nx,ny):
	const_mult = alpha*dt/(dx*dx)

	u[1:nx + 1, 1:ny + 1] = utemp[1:nx + 1, 1:ny + 1] + const_mult * (
				-4.0 * utemp[1:nx + 1, 1:ny + 1] + (utemp[0:nx, 1:ny + 1] +
													utemp[2:nx + 2, 1:ny + 1] + utemp[1:nx + 1, 0:ny] + utemp[1:nx + 1,
																										2:ny + 2]))

	utemp[:,:] = u[:,:]

if __name__ == "__main__":
	#MPI fluff
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nprocs = comm.Get_size()

	nx = int(nx_global // np.sqrt(nprocs))
	ny = nx

	# Initial conditions
	x_local, y_local, u_local = initiate_local_domain(rank,nprocs)
	utemp = np.copy(u_local)

	if rank == 0:
		t1 = time.time()

	t = 0.0
	while t < ft:
		t = t + dt

		ftcs_update(u_local,utemp,nx,ny)

		exchange_boundary_information(u_local, comm, rank, nprocs)



	if rank == 0:
		print('CPU time taken for MPI + Vectorized FTCS: ', time.time() - t1)

	write_field(u_local,rank,nprocs)

	if rank == 0:
		plot_field(x_local,y_local,u_local,'Check')
		print(np.shape(u_local))
