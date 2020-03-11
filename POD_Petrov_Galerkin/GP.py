import numpy as np
import autograd.numpy as anp
from autograd import grad, jacobian

from Parameters import tsteps, Rnum, dx, dt, seq_num, spatial_resolution
from Problem import nl_calc, l_calc

import matplotlib.pyplot as plt


class gp_evolution():
    '''
    Does a POD/POD-DEIM GP evolution
    '''
    def __init__(self,V,U,P,ytilde_init):
        '''
        ytilde - initial condition in POD space - shape: Kx1
        V - Truncated POD bases (precomputed) - shape: NxK
        V - Truncated POD bases for nonlinear snapshots (precomputed) - shape: NxM
        P - DEIM matrix - shape: NxM
        '''
        self.V = V
        self.U = U
        self.P = P
        self.num_steps = int(np.shape(tsteps)[0]-1)
        self.state_tracker = np.zeros(shape=(np.shape(ytilde_init)[0],self.num_steps+1),dtype='double')
        self.state_tracker[:,0] = ytilde_init[:,0]
        self.nl_state_tracker = np.zeros(shape=(np.shape(ytilde_init)[0],self.num_steps+1),dtype='double')

    def linear_operator_fixed(self):
        '''
        This method fixes the laplacian based linear operator
        '''
        N_ = np.shape(self.V)[0]
        self.laplacian_matrix = -2.0*np.identity(N_)

        # Setting upper diagonal
        i_range = np.arange(0,N_-1,dtype='int')
        j_range = np.arange(1,N_,dtype='int')

        self.laplacian_matrix[i_range,j_range] = 1.0

        # Setting lower diagonal
        i_range = np.arange(1,N_,dtype='int')
        j_range = np.arange(0,N_-1,dtype='int')

        self.laplacian_matrix[i_range,j_range] = 1.0

        # Periodicity
        self.laplacian_matrix[0,N_-1] = 1.0
        self.laplacian_matrix[N_-1,0] = 1.0
        self.laplacian_matrix = 1.0/(Rnum*dx*dx)*self.laplacian_matrix
        
        # Final Linear operator
        self.linear_matrix = np.matmul(np.matmul(np.transpose(self.V),self.laplacian_matrix),self.V)

    def linear_operator(self,Ytilde):
        '''
        Calculates the linear term on the RHS
        '''
        M_ = np.shape(self.V)[1]
        return np.matmul(self.linear_matrix,Ytilde).reshape(M_,1)

    def nonlinear_operator_pod(self,ytilde):
        '''
        Defines the nonlinear reconstruction using the standard POD-GP technique
        '''
        N_ = np.shape(self.V)[0]
        field = np.matmul(self.V,ytilde).reshape(N_,1)
        nonlinear_field_approx = nl_calc(field)
        return np.matmul(np.transpose(self.V),nonlinear_field_approx)

    def pod_gp_rhs(self,state):
        '''
        Calculate the rhs of the POD GP implementation
        '''
        linear_term = self.linear_operator(state)
        non_linear_term = self.nonlinear_operator_pod(state)
    
        return np.add(linear_term,-non_linear_term)

    def pod_gp_evolve(self):
        '''
        Use RK3 to do a system evolution for pod_gp
        '''
        # Setup fixed operations
        self.linear_operator_fixed()
        state = np.copy(self.state_tracker[:,0])
        
        # Recording the nonlinear term
        non_linear_term = self.nonlinear_operator_pod(state)
        self.nl_state_tracker[:,0] = non_linear_term[:,0]

        for t in range(1,self.num_steps+1):
            
            rhs = self.pod_gp_rhs(state)
            l1 = state + dt*rhs[:,0]

            rhs = self.pod_gp_rhs(l1)
            l2 = 0.75*state + 0.25*l1 + 0.25*dt*rhs[:,0]

            rhs = self.pod_gp_rhs(l2)
            state[:] = 1.0/3.0*state[:] + 2.0/3.0*l2[:] + 2.0/3.0*dt*rhs[:,0]

            self.state_tracker[:,t] = state[:]

            # Recording the nonlinear term
            non_linear_term = self.nonlinear_operator_pod(state)
            self.nl_state_tracker[:,t] = non_linear_term[:,0]

    def deim_matrices_precompute(self):
        '''
        Precompute the DEIM matrices for nonlinear term in ODE
        '''
        mtemp = np.linalg.pinv(np.matmul(np.transpose(self.P),self.U)) #ok MxM
        mtemp = np.matmul(self.U,mtemp) # NxM
        mtemp = np.matmul(mtemp,np.transpose(self.P)) # NxN
        
        self.varP = np.matmul(np.transpose(self.V),mtemp) # KxN

    def nonlinear_operator_pod_deim(self,ytilde):
        '''
        Defines the nonlinear reconstruction using the POD-DEIM technique
        '''
        N_ = np.shape(self.V)[0]
        nl_calc_locs = np.asarray(self.varP[0].nonzero()) # Nonlinear term should only be computed at these locations

        field = np.matmul(self.V,ytilde).reshape(N_,1) # Linear operation
        nonlinear_field_approx = self.nl_calc_deim(field,nl_calc_locs)
        return np.matmul(self.varP,nonlinear_field_approx)


    def nl_calc_deim(self,u,nl_calc_locs):
        '''
        Field u - shape: Nx1
        Returns udu/dx - shape : Nx1
        '''
        # Find neighbors of calculation locations
        nl_locs_left = nl_calc_locs-1
        nl_locs_right = nl_calc_locs+1

        nl_locs_left = np.where(nl_locs_left>=0,nl_locs_left,nl_locs_left+spatial_resolution)
        nl_locs_right = np.where(nl_locs_right<=spatial_resolution-1,nl_locs_right,nl_locs_right-spatial_resolution)


        temp_array = np.zeros(shape=(np.shape(u)[0]+2,1)) # Temp array
        unl = np.zeros(shape=(np.shape(u)[0],1)) # Array for calculating the nonlinear term

        unl[nl_calc_locs,0] = u[nl_calc_locs,0]*(u[nl_locs_right,0]-u[nl_locs_left,0])/(2.0*dx)

        return unl

    def pod_deim_rhs(self,state):
        '''
        Calculate the rhs of the POD GP implementation
        '''
        linear_term = self.linear_operator(state)
        non_linear_term = self.nonlinear_operator_pod_deim(state)
    
        return np.add(linear_term,-non_linear_term)

    def pod_deim_evolve(self):
        '''
        Use RK3 to do a system evolution for pod_deim
        '''
        # Setup fixed operations
        self.linear_operator_fixed()
        self.deim_matrices_precompute()
        state = np.copy(self.state_tracker[:,0])
        
        # Recording the nonlinear term
        non_linear_term = self.nonlinear_operator_pod_deim(state)
        self.nl_state_tracker[:,0] = non_linear_term[:,0]
        
        for t in range(1,self.num_steps+1):
            
            rhs = self.pod_deim_rhs(state)
            l1 = state + dt*rhs[:,0]

            rhs = self.pod_deim_rhs(l1)
            l2 = 0.75*state + 0.25*l1 + 0.25*dt*rhs[:,0]

            rhs = self.pod_deim_rhs(l2)
            state[:] = 1.0/3.0*state[:] + 2.0/3.0*l2[:] + 2.0/3.0*dt*rhs[:,0]

            self.state_tracker[:,t] = state[:]

            # Recording the nonlinear term
            non_linear_term = self.nonlinear_operator_pod_deim(state)
            self.nl_state_tracker[:,t] = non_linear_term[:,0]

class pgp_evolution():
    '''
    Does a POD Petrov-Galerkin evolution (may or may not include deim)
    '''
    def __init__(self,V,U,P,ytilde_init):
        '''
        ytilde - initial condition in POD space - shape: Kx1
        V - Truncated POD bases (precomputed) - shape: NxK
        V - Truncated POD bases for nonlinear snapshots (precomputed) - shape: NxM
        P - DEIM matrix - shape: NxM
        '''
        self.V = V
        self.U = U
        self.P = P
        self.num_steps = int(np.shape(tsteps)[0]-1)
        self.state_tracker = np.zeros(shape=(np.shape(ytilde_init)[0],self.num_steps+1),dtype='double')
        self.state_tracker[:,0] = ytilde_init[:,0]
        self.nl_state_tracker = np.zeros(shape=(np.shape(ytilde_init)[0],self.num_steps+1),dtype='double')


    def linear_operator_fixed(self):
        '''
        This method fixes the laplacian based linear operator
        '''
        N_ = np.shape(self.V)[0]
        self.laplacian_matrix = -2.0*np.identity(N_)

        # Setting upper diagonal
        i_range = np.arange(0,N_-1,dtype='int')
        j_range = np.arange(1,N_,dtype='int')

        self.laplacian_matrix[i_range,j_range] = 1.0

        # Setting lower diagonal
        i_range = np.arange(1,N_,dtype='int')
        j_range = np.arange(0,N_-1,dtype='int')

        self.laplacian_matrix[i_range,j_range] = 1.0

        # Periodicity
        self.laplacian_matrix[0,N_-1] = 1.0
        self.laplacian_matrix[N_-1,0] = 1.0
        self.laplacian_matrix = 1.0/(Rnum*dx*dx)*self.laplacian_matrix
        
        # Final Linear operator
        self.linear_matrix = np.matmul(np.matmul(np.transpose(self.V),self.laplacian_matrix),self.V)

    def linear_operator(self,ytilde):
        '''
        Calculates the linear term on the RHS
        '''
        M_ = np.shape(self.V)[1]
        return np.matmul(self.linear_matrix,ytilde).reshape(M_,1)


    def deim_matrices_precompute(self):
        '''
        Precompute the DEIM matrices for nonlinear term in ODE
        '''
        mtemp = np.linalg.pinv(np.matmul(np.transpose(self.P),self.U)) #ok MxM
        mtemp = np.matmul(self.U,mtemp) # NxM
        mtemp = np.matmul(mtemp,np.transpose(self.P)) # NxN
        
        self.varP = np.matmul(np.transpose(self.V),mtemp) # KxN

    def nonlinear_operator_pod_deim(self,ytilde):
        '''
        Defines the nonlinear reconstruction using the POD-DEIM technique
        '''
        N_ = np.shape(self.V)[0]
        nl_calc_locs = np.asarray(self.varP[0].nonzero()) # Nonlinear term should only be computed at these locations

        field = np.matmul(self.V,ytilde).reshape(N_,1) # Linear operation
        nonlinear_field_approx = self.nl_calc_deim(field,nl_calc_locs)
        return np.matmul(self.varP,nonlinear_field_approx)


    def nl_calc_deim(self,u,nl_calc_locs):
        '''
        Field u - shape: Nx1
        Returns udu/dx - shape : Nx1
        '''
        # Find neighbors of calculation locations
        nl_locs_left = nl_calc_locs-1
        nl_locs_right = nl_calc_locs+1

        nl_locs_left = np.where(nl_locs_left>=0,nl_locs_left,nl_locs_left+spatial_resolution)
        nl_locs_right = np.where(nl_locs_right<=spatial_resolution-1,nl_locs_right,nl_locs_right-spatial_resolution)


        temp_array = np.zeros(shape=(np.shape(u)[0]+2,1)) # Temp array
        unl = np.zeros(shape=(np.shape(u)[0],1)) # Array for calculating the nonlinear term

        unl[nl_calc_locs,0] = u[nl_calc_locs,0]*(u[nl_locs_right,0]-u[nl_locs_left,0])/(2.0*dx)

        return unl

    def nonlinear_operator_pod(self,ytilde):
        '''
        Defines the nonlinear reconstruction using the standard POD-GP technique
        '''
        N_ = np.shape(self.V)[0]
        field = np.matmul(self.V,ytilde).reshape(N_,1)
        nonlinear_field_approx = nl_calc(field)
        return np.matmul(np.transpose(self.V),nonlinear_field_approx)

    def reconstruct(self,ytilde):
        N_ = np.shape(self.V)[0]
        field = np.matmul(self.V,ytilde).reshape(N_,1)
        return field

    def residual_fo(self,u,u_new):
        '''
        Defines the residual to be minimized by implicit time integration
        Need to define as a function of the full state
        '''
        return (u_new[:,0] - u[:,0])/dt -(l_calc(u_new)[:,0]) + nl_calc(u_new)[:,0]

    def test_space(self,ytilde,ytilde_new):
        u = self.reconstruct(ytilde)
        u_new = self.reconstruct(ytilde_new)

        jacobian_res = jacobian(self.residual_fo)
        return np.transpose(np.matmul(jacobian_res(u,u_new)[:,:,0],self.V))

    def residual_ro(self,ytilde_new,ytilde):
        # Get test space
        wt = self.test_space(ytilde,ytilde_new)

        # Get reduced-order residual
        u = self.reconstruct(ytilde)
        u_new = self.reconstruct(ytilde_new)
        r = self.residual_fo(u,u_new)

        # Project residual on test space
        return np.sum(np.abs(np.matmul(wt,r)))

    def pod_pgp_evolve(self):
        from scipy.optimize import minimize
        '''
        Euler implicit for evolution
        '''
        # Setup fixed operations
        # self.linear_operator_fixed()
        state = np.copy(self.state_tracker[:,0])
        state_new = np.zeros_like(state)

        self.deim_matrices_precompute()
        
        # Recording the nonlinear term
        non_linear_term = self.nonlinear_operator_pod_deim(state)
        self.nl_state_tracker[:,0] = non_linear_term[:,0]

        plt_iter = 0
        plt.figure()
        plt.plot(self.reconstruct(state),label='0')
       
        for t in range(1,self.num_steps+1):          

            print('initial residual:',self.residual_ro(state_new,state))
            state_new = np.copy(minimize(self.residual_ro,state,args=(state),method='Nelder-Mead', tol=1e-6).x)
            print('Final residual:',self.residual_ro(state_new,state))

            if plt_iter == 10:
                plt.plot(self.reconstruct(state_new),label=str(t))
                plt_iter = 0

            state = np.copy(state_new)
            self.state_tracker[:,t] = state[:]

            # Recording the nonlinear term
            non_linear_term = self.nonlinear_operator_pod_deim(state)
            self.nl_state_tracker[:,t] = non_linear_term[:,0]

            plt_iter = plt_iter + 1
            print('Time is: ',t)

        plt.legend()
        plt.show()