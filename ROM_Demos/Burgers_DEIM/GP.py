import numpy as np
from Parameters import tsteps, Rnum, dx, dt, seq_num
from Problem import nl_calc

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
        field = np.matmul(self.V,ytilde).reshape(N_,1)
        nonlinear_field_approx = nl_calc(field)
        return np.matmul(self.varP,nonlinear_field_approx)

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

    def slfn_gp_evolve(self,model):
        '''
        Use Euler forward to do a system evolution for slfn_gp
        '''
        # Setup fixed operations
        self.linear_operator_fixed()
        self.deim_matrices_precompute()
        state = np.copy(self.state_tracker[:,0])
        N_ = np.shape(state)[0]
        non_linear_term = self.nonlinear_operator_pod_deim(state)
        for t in range(1,self.num_steps+1):
            
            linear_term = self.linear_operator(state)
            non_linear_term = np.transpose(model.predict(non_linear_term.reshape(1,N_)))
            rhs = np.add(linear_term,-non_linear_term.reshape(N_,1))
            state[:] = state[:] + dt*rhs[:,0]

            self.state_tracker[:,t] = state[:]

    def lstm_gp_evolve(self,model):
        '''
        Use Euler forward to do a system evolution for slfn_gp
        '''
        # Setup fixed operations
        state = np.copy(self.state_tracker[:,0])
        self.state_tracker[:,1:] = 0.0
        self.linear_operator_fixed()
        self.deim_matrices_precompute()
        N_ = np.shape(state)[0]
        non_linear_term_input = np.zeros(shape=(1,seq_num,N_))

        for t in range(0,seq_num):
            rhs = self.pod_deim_rhs(state)
            l1 = state + dt*rhs[:,0]

            rhs = self.pod_deim_rhs(l1)
            l2 = 0.75*state + 0.25*l1 + 0.25*dt*rhs[:,0]

            rhs = self.pod_deim_rhs(l2)
            state[:] = 1.0/3.0*state[:] + 2.0/3.0*l2[:] + 2.0/3.0*dt*rhs[:,0]
            self.state_tracker[:,t] = state[:]

            non_linear_term_input[0,t,:] = self.nonlinear_operator_pod_deim(state)[:,0]
       
        for t in range(seq_num,self.num_steps+1):
            linear_term = self.linear_operator(state)
            non_linear_term_next = np.transpose(model.predict(non_linear_term_input))

            rhs = np.add(linear_term,-non_linear_term_next.reshape(N_,1))
            
            state[:] = state[:] + dt*rhs[:,0]
            self.state_tracker[:,t] = state[:]

            non_linear_term_input[0,:-1,:] = non_linear_term_input[0,1:,:]
            non_linear_term_input[0,-1,:] = non_linear_term_next[:,0]







    


