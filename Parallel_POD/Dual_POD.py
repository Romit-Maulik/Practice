import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

# Make sure you have this to ensure numpy doesn't automatically use multiple threads on a single compute node
# export OPENBLAS_NUM_THREADS=1

# Number of modes to keep (can vary across rank but simple for now)
num_modes = 12

# Method of snapshots to accelerate
def generate_pod_bases(Y): #Mean removed
    '''
    Y - Snapshot matrix - shape: NxS
    returns V - truncated POD basis matrix - shape: NxK
    '''
    new_mat = np.matmul(np.transpose(Y),Y)
    w,v = np.linalg.eig(new_mat)

    # Bases
    V = np.real(np.matmul(Y,v)) 
    trange = np.arange(np.shape(V)[1])
    V[:,trange] = V[:,trange]/np.sqrt(w[:])

    # Truncate phis
    V = V[:,:num_modes] # Columns are modes

    # Alternatively
    # # Run a local SVD and threshold
    # ulocal, slocal, vlocal = np.linalg.svd(local_data)

    # # Local POD basis
    # V = ulocal[:,:num_modes]

    return V

if __name__ == '__main__':
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # Check what data you have to grab
    # Here we assume that the snapshots are already segregated into different files
    # Should be (points) x (snapshots per rank) - total data matrix is points x nprocs*snapshots
    local_data = np.load('snapshots_rank_'+str(rank)+'.npy')

    ulocal = generate_pod_bases(local_data)

    # Gather on rank zero
    uglobal = comm.gather(ulocal,root=0)

    if rank == 0:
        # Do some reshaping
        uglobal = np.asarray(uglobal)
        uglobal = np.rollaxis(uglobal,axis=1)
        uglobal = np.rollaxis(uglobal,2,1)
        uglobal = np.asarray(uglobal).reshape(-1,num_modes*nprocs)
        
        u = generate_pod_bases(uglobal)

        # calculate final POD basis
        u = u[:,:num_modes]

        # Save
        np.save('Dual_POD_Basis.npy',u)