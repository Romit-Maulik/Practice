import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

# Make sure you have this to ensure numpy doesn't automatically use multiple threads on a single compute node
# export OPENBLAS_NUM_THREADS=1

# Number of modes to keep (can vary across rank but simple for now)
num_modes = 12

# Method of snapshots to accelerate
def generate_right_vectors(Y): #Mean removed
    '''
    Y - Snapshot matrix - shape: NxS
    returns V - truncated right singular vectors
    '''
    new_mat = np.matmul(np.transpose(Y),Y)
    w,v = np.linalg.eig(new_mat)

    return v[:,:num_modes], np.sqrt(w[:num_modes])

if __name__ == '__main__':
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # Check what data you have to grab
    # Here we assume that the snapshots are already segregated into different files
    # Should be (points per rank) x (snapshots) - total data matrix is nprocs*points x snapshots
    local_data = np.load('points_rank_'+str(rank)+'.npy') 

    # Run a local SVD and threshold
    vlocal, slocal = generate_right_vectors(local_data)

    # Find W
    wlocal = np.matmul(vlocal,np.diag(slocal).T)

    # Gather data at rank 0:
    wglobal = comm.gather(wlocal,root=0)

    # perform SVD at rank 0:
    if rank == 0:
        wglobal = np.asarray(wglobal)
        wglobal = np.rollaxis(wglobal,axis=1)
        wglobal = np.rollaxis(wglobal,axis=2,start=1)
        wglobal = np.asarray(wglobal).reshape(-1,num_modes*nprocs)
        x, s, y = np.linalg.svd(wglobal)
    else:
        x = None
        s = None
    
    x = comm.bcast(x,root=0)
    s = comm.bcast(s,root=0)
    
    # perform APMOS at each local rank
    phi_local = []
    for mode in range(num_modes):
        phi_temp = 1.0/np.sqrt(s[mode])*np.matmul(local_data,x[:,mode:mode+1])
        phi_local.append(phi_temp)

    phi_local = np.asarray(phi_local)

    # Gather modes at rank 0
    phi_global = comm.gather(phi_local[:,:,0].T,root=0)

    if rank == 0:
        np.save('APMOS_Basis.npy',np.asarray(phi_global).reshape(-1,num_modes))









