import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

spatial_resolution = 512 # Grid points
temporal_resolution = 100 # Snapshots
final_time = 2.0 # Keep fixed

# Independent variables
x = np.linspace(0.0,1.0,num=spatial_resolution)
tsteps = np.linspace(0.0,2.0,num=temporal_resolution)

# Truncation
num_modes = 20

def generate_pod_bases(Y): #Mean removed
    '''
    Y - Snapshot matrix - shape: NxS
    returns V - truncated POD basis matrix - shape: NxS
    '''
    new_mat = np.matmul(np.transpose(Y),Y)
    w,v = np.linalg.eig(new_mat)
    w = np.abs(w)

    # Bases
    V = np.real(np.matmul(Y,v)) 
    trange = np.arange(np.shape(V)[1])
    V[:,trange] = V[:,trange]/np.sqrt(w[:])

    return V[:,:num_modes]

def exact_solution_field(Rnum,t):
    t0 = np.exp(Rnum/8.0)
    return (x/(t+1))/(1.0+np.sqrt((t+1)/t0)*np.exp(Rnum*(x*x)/(4.0*t+4)))

def compute_pod(Rnum):
    snapshot_matrix_total = np.zeros(shape=(np.shape(x)[0],np.shape(tsteps)[0]))

    trange = np.arange(np.shape(tsteps)[0])
    for t in trange:
        snapshot_matrix_total[:,t] = exact_solution_field(Rnum,tsteps[t])[:]

    snapshot_matrix_mean = np.mean(snapshot_matrix_total,axis=1)
    snapshot_matrix = (snapshot_matrix_total.transpose()-snapshot_matrix_mean).transpose()

    V = generate_pod_bases(snapshot_matrix)

    return V

if __name__ == '__main__':

    param_vals = np.arange(100,1000,100)


    # Find the interpolation points
    gamma_list = []
    for i in range(np.shape(param_vals)[0]):
        param = param_vals[i]
        V = compute_pod(param)
        idim = V.shape[0]

        if i == 0:

            V0V0_T = np.matmul(V,V.T)
            lmat = np.eye(idim)-V0V0_T
            V0 = V.copy()


        rmat = np.linalg.inv(np.matmul(V0.T,V))
        mat_temp = np.linalg.multi_dot([lmat,V,rmat])
        u, s, vt = np.linalg.svd(mat_temp, full_matrices=False)

        gamma = np.linalg.multi_dot([u,np.arctan(np.diag(s)),vt])
        gamma_list.append(gamma)


    # Perform interpolation at new location
    new_param = 250
    new_mat = np.zeros_like(gamma_list[0])

    for i in range(new_mat.shape[0]):
        for j in range(new_mat.shape[1]):

            int_points = []
            for gamma in gamma_list:
                int_points.append(gamma[i,j])

            cs = CubicSpline(param_vals, np.asarray(int_points))
            new_mat[i,j] = cs(new_param)


    # Singular value decomposition
    u, s, vt = np.linalg.svd(new_mat, full_matrices=False)
    new_basis = np.linalg.multi_dot([V0,vt.T,np.cos(np.diag(s))]) \
                + np.linalg.multi_dot([u,np.sin(np.diag(s))])

    plt.figure()
    plt.plot(new_basis[:,0],label='Mode 1')
    plt.plot(new_basis[:,1],label='Mode 2')
    plt.plot(new_basis[:,2],label='Mode 3')
    plt.legend()
    plt.show()

