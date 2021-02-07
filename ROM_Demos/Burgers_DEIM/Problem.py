#-------------------------------------------------------------------------------------------------
# This is the Burgers problem definition file
#-------------------------------------------------------------------------------------------------
import numpy as np
from Parameters import Rnum, tsteps, x, dx

def exact_solution_field(t):
    t0 = np.exp(Rnum/8.0)
    return (x/(t+1))/(1.0+np.sqrt((t+1)/t0)*np.exp(Rnum*(x*x)/(4.0*t+4)))

def exact_solution_nonlinear(t):
    t0 = np.exp(Rnum/8.0)
    aval = np.sqrt((t+1)/t0)
    bval = Rnum/(4.0*t+4)
    cval = 1/(t+1)

    u = (cval*x)/(1.0+aval*np.exp(bval*x*x))
    dudx = -cval/((aval*np.exp(bval*x*x)+1)**2)*(2.0*aval*bval*x*x*np.exp(bval*x*x) - aval*np.exp(bval*x*x) - 1)
    nl = u*dudx
    
    return nl

def collect_snapshots_field():
    snapshot_matrix_total = np.zeros(shape=(np.shape(x)[0],np.shape(tsteps)[0]))

    trange = np.arange(np.shape(tsteps)[0])
    for t in trange:
        snapshot_matrix_total[:,t] = exact_solution_field(tsteps[t])[:]

    snapshot_matrix_mean = np.mean(snapshot_matrix_total,axis=1)
    snapshot_matrix = (snapshot_matrix_total.transpose()-snapshot_matrix_mean).transpose()

    return snapshot_matrix, snapshot_matrix_mean, snapshot_matrix_total

def collect_snapshots_nonlinear():
    snapshot_matrix_total = np.zeros(shape=(np.shape(x)[0],np.shape(tsteps)[0]))

    trange = np.arange(np.shape(tsteps)[0])
    for t in trange:
        snapshot_matrix_total[:,t] = exact_solution_nonlinear(tsteps[t])[:]

    snapshot_matrix_mean = np.mean(snapshot_matrix_total,axis=1)
    snapshot_matrix = (snapshot_matrix_total.transpose()-snapshot_matrix_mean).transpose()

    return snapshot_matrix, snapshot_matrix_mean, snapshot_matrix_total

def nl_calc(u):
    '''
    Field u - shape: Nx1
    Returns udu/dx - shape : Nx1
    '''
    temp_array = np.zeros(shape=(np.shape(u)[0]+2,1)) # Temp array
    unl = np.zeros(shape=(np.shape(u)[0],1)) # Array for calculating the nonlinear term

    # Copy the field
    temp_array[1:-1,0] = u[:,0]
    # Periodicity
    temp_array[0,0] = u[-1,0]
    temp_array[-1,0] = u[0,0]

    unl[:,0] = u[:,0]*(temp_array[2:,0]-temp_array[0:-2,0])/(2.0*dx)

    return unl

if __name__ == "__main__":
    print('Problem definition file')