import numpy as np
import matplotlib.pyplot as plt
from Parameters import fac, gappy
from Problem import nl_calc

def field_compression(Y,K):
    '''
    Y - Snapshot matrix - shape: NxS
    K - number of modes to truncate to
    returns 
    V - truncated POD basis matrix - shape: NxK
    Ytilde - truncated coefficient matrix - shape: KxS
    '''
    V = generate_pod_bases(Y,K)
    Ytilde = field_coefficients(Y,V)

    return V, Ytilde

def generate_pod_bases(Y,K): #Mean removed
    '''
    Y - Snapshot matrix - shape: NxS
    returns V - truncated POD basis matrix - shape: NxK
    '''
    new_mat = np.matmul(np.transpose(Y),Y)
    w,v = np.linalg.eig(new_mat)

    # plt.figure()
    # plt.semilogy(w[:]/np.sum(w)*100)
    # plt.show()

    # Bases
    V = np.real(np.matmul(Y,v)) 
    trange = np.arange(np.shape(V)[1])
    V[:,trange] = V[:,trange]/np.sqrt(w[:])

    # Truncate phis
    V = V[:,0:K] # Columns are modes

    return V

def field_coefficients(Y,V):
    '''
    Y - Snapshot matrix - shape: NxS
    V - truncated POD basis matrix - shape: NxK
    returns Ytilde - shape: KxS
    '''
    return np.matmul(np.transpose(V),Y)

def nonlinear_compression(V,F,M):
    '''
    V - POD bases for field snapshots - shape: NxK
    F - Nonlinear term snapshots: NxS
    P - DEIM matrix
    returns
    U - POD bases for nonlinear term snapshots: NxM
    Ftilde - DEIM coefficients: KxS
    '''
    U = generate_pod_bases(F,M)

    if gappy == True:
        P = gappy_deim_matrix(U)
    else:
        P = deim_matrix(U)

    Ftilde = deim_coefficients(V,U,F,P)

    return U, Ftilde, P

def deim_matrix(U):#U are the nonlinear snapshot pod modes
    '''
    U - POD bases for nonlinear term snapshots - shape: NxM
    returns P - DEIM matrix - shape - NxM
    '''
    p = np.zeros(shape=(1,np.shape(U)[1]),dtype='int')  
    p[0,0] = np.argmax(np.abs(U[:,0]))
    utemp = U[:,0].reshape(np.shape(U)[0],1)
    id_mat = np.identity(np.shape(U)[0])
    P = id_mat[:,p[0,0]].reshape(np.shape(U)[0],1)

    for ii in range(1,np.shape(U)[1]):
        inv_mat = np.linalg.inv(np.matmul(np.transpose(P),utemp))
        rhs_mat = np.matmul(np.transpose(P),U[:,ii].reshape(np.shape(U)[0],1))
        c = np.matmul(inv_mat,rhs_mat)

        rvec = U[:,ii].reshape(np.shape(U)[0],1)-np.matmul(utemp,c)
        p[0,ii] = np.argmax(np.abs(rvec))

        utemp = np.concatenate((utemp,U[:,ii].reshape(np.shape(U)[0],1)),axis=1)
        P = np.concatenate((P,id_mat[:,p[0,ii]].reshape(np.shape(U)[0],1)),axis=1)

    return P

def deim_coefficients(V,U,F,P):
    '''
    V - POD bases for field snapshots : NxK
    U - POD bases for nonlinear term snapshots : NxM
    F - Nonlinear term snapshots : NxS
    P - DEIM matrix : NxM
    '''
    mid_mat = np.linalg.pinv(np.matmul(np.transpose(P),U)) #ok MxM
    l_mat = np.matmul(U,mid_mat) # NxM
    varP = np.matmul(l_mat,np.transpose(P)) # NxN
    Ftilde = np.matmul(np.matmul(np.transpose(V),varP),F) # KxS

    return Ftilde

def plot_pod_modes(phi,mode_num):
    plt.figure()
    plt.plot(phi[:,mode_num])
    plt.show()

def gappy_deim_matrix(U):
    N_ = np.shape(U)[0]
    M_ = np.shape(U)[1]

    p = np.zeros(shape=(fac,M_),dtype='int')
    NodesUnused = np.arange(N_,dtype='int')

    idx = np.argsort(U[:,0],axis=0)[::-1] # Sorting in descending order with indices
    p[:,0] = idx[0:fac]

    id_mat = np.identity(N_)
    utemp = np.copy(U)

    id_mat[p[:,0],0] = 1
    P = np.zeros(shape=(N_,fac*M_))
    P[:,0:fac] = id_mat[:,p[:,0]]

    NodesUnused = np.setdiff1d(NodesUnused,p[:,0])

    for ii in range(1,M_):
        m1 = P[:,0:ii*fac]
        m2 = utemp[:,:ii]
        m3 = np.linalg.pinv(np.matmul(np.transpose(m1),m2))
        m4 = np.matmul(m3,np.transpose(m1))
        c_ = np.matmul(m4,U[:,ii].reshape(N_,1))
        
        rvec = U[:,ii].reshape(N_,1)-np.matmul(utemp[:,0:ii].reshape(N_,1),c_)

        idx = np.argsort(np.abs(rvec),axis=0)[::-1] # Sorting in descending order with indices
        membership = np.isin(idx,NodesUnused)

        count = 0
        iter_val = 0

        while count < fac and iter_val < np.shape(membership)[0]:
            if membership[iter_val] == True:
                p[count,ii] = idx[iter_val]
                count = count + 1
                NodesUnused = np.setdiff1d(NodesUnused,idx[iter_val])
                iter_val = iter_val + 1
            else:
                iter_val = iter_val + 1
                   
        P[:,ii*fac:(ii+1)*fac] = id_mat[:,p[:,ii]]

        return P

def nl_reconstruct(V,P,U,Ytilde_S):
    '''
    V - POD basis matrix for field - shape: NxK
    U - POD basis matrix for nonlinear term - shape : NxM
    Ytilde_S - shape: Kx1
    P - DEIM matrix - shape: NxM
    '''
    # Full order field reconstruction
    ufield = np.matmul(V,Ytilde_S)
    # Calculate the nonlinear term given ufield
    unl = nl_calc(ufield)
    # Calculate DEIM coefficient (on the fly)
    mid_mat = np.linalg.pinv(np.matmul(np.transpose(P),U)) #ok MxM
    mid_mat = np.matmul(U,mid_mat) # NxM
    mid_mat = np.matmul(mid_mat,np.transpose(P)) # NxN
    mid_mat = np.matmul(np.transpose(V),mid_mat) # KxN
    Ftilde_S = np.matmul(mid_mat,unl) # Kx1
    
    return Ftilde_S

if __name__ == "__main__":
    print('Data compression file')
