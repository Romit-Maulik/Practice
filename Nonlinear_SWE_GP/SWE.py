import numpy as np
np.random.seed(10)
from problem import shallow_water, shallow_water_rom, plot_coefficients, plot_coefficients_compare
from parameters import fvm_solve

lhc_sampling = False

if __name__ == '__main__':
    if fvm_solve:

        print('Running non-linear SWE')

        if lhc_sampling:
            locs = np.array([[-0.45, -0.25],
                           [-0.15,  0.35],
                           [-0.05, -0.45],
                           [ 0.15,  0.45],
                           [ 0.05,  0.15],
                           [-0.35,  0.05],
                           [ 0.35, -0.05],
                           [-0.25, -0.15],
                           [ 0.45,  0.25],
                           [ 0.25, -0.35]])

            for loc_num in range(np.shape(locs)[0]):
                filename = 'snapshot_matrix_pod_'+str(loc_num+1)+'.npy'

                new_run = shallow_water(locs[loc_num])
                new_run.solve()
        else:
            new_run = shallow_water([-1.0/2.7, -1.0/4.0])
            new_run.solve()

        print('Saving snapshot data for POD')
        snapshot_matrix_pod = np.transpose(np.array(new_run.snapshots_pod))
        print('Shape of snapshot matrix for POD:',np.shape(snapshot_matrix_pod))
        np.save('snapshot_matrix_pod.npy',snapshot_matrix_pod)

    else:
        # Loading snapshots
        snapshot_matrix_pod = np.load('snapshot_matrix_pod.npy')
        # Shape
        print('Shape of the snapshot matrix for each pod:',np.shape(snapshot_matrix_pod))
        
        # Initialize ROM class
        gprom = shallow_water_rom(snapshot_matrix_pod)
        # Find POD and DEIM coefficients
        gprom.generate_pod()
        # gprom.plot_reconstruction_error()      
        # Do GP solve using equations
        gprom.solve()



    