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

    plt.figure()
    plt.plot(xi,c_vec*S_up,color='blue')
    plt.plot(np.flip(xi),np.flip(c_vec*S_down),color='blue')
    plt.ylim((-0.5,0.5))
    plt.savefig('Shapes/Shape_'+str(pnum)+'.jpeg')
    plt.close()

if __name__ == '__main__':
    pass
