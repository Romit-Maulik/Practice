import matplotlib.pyplot as plt
import numpy as np
np.random.seed(10)
import glob

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(dir_path+'/Shapes/'):
    os.mkdir(dir_path+'/Shapes')
else:
    files = glob.glob(dir_path+'/Shapes/*.jpeg')
    for f in files:
        os.remove(f)

# Coefficient of determination
def coeff_determination(y_pred, y_true): #Order of function inputs is important here        
    SS_res =  np.sum(np.square( y_true-y_pred )) 
    SS_tot = np.sum(np.square( y_true - np.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + 2.22044604925e-16) )

# Shape of airfoil
def shape_return(a,pnum):
    x0 = 0.0
    x1 = 1.0
    p = 130

    xi = np.linspace(x0,x1,num=p)
    c_vec = np.sqrt(xi)*(x1-xi)
    a = a.flatten()
    N = np.shape(a)[0]//2

    S_up = 0
    S_down = 0
    
    for i in range(N):
        S_up = S_up + a[i]*(xi**i)*(x1 - xi)**(N-i)
        
    for i in range(N):
        S_down = S_down + a[N+i]*(xi**i)*(x1 - xi)**(N-i)

    xx = np.concatenate((np.flip(xi),xi))
    cc = np.concatenate((np.flip(c_vec*S_up),c_vec*S_down))

    plt.figure()
    plt.plot(xx,cc,label='Found')
    plt.ylim((-0.1,0.1))

    from constraints import t_base
    
    t_base = np.asarray(t_base)
    t_base = t_base.flatten()
    N = np.shape(t_base)[0]//2

    S_up = 0
    S_down = 0
    
    for i in range(N):
        S_up = S_up + t_base[i]*(xi**i)*(x1 - xi)**(N-i)
        
    for i in range(N):
        S_down = S_down + t_base[N+i]*(xi**i)*(x1 - xi)**(N-i)

    cc = np.concatenate((np.flip(c_vec*S_up),c_vec*S_down))
    plt.plot(xx,cc,label='Base')
    plt.legend()
    plt.savefig('Shapes/Shape_'+str(pnum)+'.jpeg')
    plt.close()

if __name__ == '__main__':
    print('Checking plotting of airfoil')
    t_base = np.asarray([0.1268, 0.467, 0.5834, 0.2103, -0.1268, -0.5425, -0.5096, 0.0581])
    t_lower = np.asarray([-0.08876, -0.3269, -0.40838, -0.14721, 0.1648, 0.70525, 0.66248, -0.04067])
    t_upper = np.asarray([0.1648, 0.6071, 0.75842, 0.27339, -0.08876, -0.37975, -0.35672, 0.07553])

    # Lower bound
    pnum = 'lower'
    shape_return(t_lower,pnum)

    # Base
    pnum = 'base'
    shape_return(t_base,pnum)

    # Upper bound
    pnum = 'upper'
    shape_return(t_upper,pnum)

