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

def plot_arrays(a,x0,x1,p):
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

    return xx, cc

# Shape of airfoil
def shape_return(a,pnum):
    x0 = 0.0
    x1 = 1.0
    p = 130

    xx, cc = plot_arrays(a,x0,x1,p)

    plt.figure()
    plt.plot(xx,cc,label='Found')
    plt.ylim((-0.1,0.1))

    from constraints import t_base   
    t_base = np.asarray(t_base)

    xx, cc = plot_arrays(t_base,x0,x1,p)
    
    plt.plot(xx,cc,label='Base')
    plt.legend()
    plt.savefig('Shapes/Shape_'+str(pnum)+'.jpeg')
    plt.close()


def shape_return_comparison(a,b,pnum):
    x0 = 0.0
    x1 = 1.0
    p = 130

    xx, cc = plot_arrays(a,x0,x1,p)
 
    plt.figure()
    plt.plot(xx,cc,label='True')

    xx, cc = plot_arrays(b,x0,x1,p)
 
    plt.plot(xx,cc,label=pnum)
    plt.ylim((-0.1,0.1))


    from constraints import t_base, t_lower, t_upper
    t_base = np.asarray(t_base)
    t_lower = np.asarray(t_lower)
    t_upper = np.asarray(t_upper)

    xx, cc = plot_arrays(t_base,x0,x1,p)
    plt.plot(xx,cc,label='Base')

    # xx, cc = plot_arrays(t_lower,x0,x1,p)
    # plt.plot(xx,cc,label='Lower bound')

    # xx, cc = plot_arrays(t_upper,x0,x1,p)
    # plt.plot(xx,cc,label='Upper bound')

    plt.legend()
    plt.savefig('Shapes/Shape_'+str(pnum)+'.jpeg')
    plt.close()

if __name__ == '__main__':
    print('Checking plotting of airfoil')
    t_base = np.asarray([0.1268, 0.467, 0.5834, 0.2103, -0.1268, -0.5425, -0.5096, 0.0581])
    t_lower = np.asarray([-0.08876, -0.3269, -0.40838, -0.14721, 0.1648, 0.70525, 0.66248, -0.04067])
    t_upper = np.asarray([0.1648, 0.6071, 0.75842, 0.27339, -0.08876, -0.37975, -0.35672, 0.07553])
    t_truth = np.asarray([0.1009, 0.3306, 0.6281, 0.1494, -0.1627, -0.6344, -0.5927, 0.0421])
    # t_dnn = np.asarray([0.08876,  0.33483461, 0.57226083, 0.16212897, -0.1648, -0.61612045, -0.53867441,  0.04067]) # 200 training data points
    t_dnn = np.asarray([0.08876, 0.33675902, 0.5756922, 0.15189275, -0.1648, -0.61284773, -0.53884293, 0.04067]) # 100 training data points
    # t_dnn = np.asarray([0.09424003, 0.33285433, 0.42598738, 0.19814173, -0.16330369, -0.69081884, -0.37384828,  0.0428294]) # with tol off and ftol on
    t_rl = np.asarray([0.0887,  0.3269,  0.40838, 0.14721, -0.1648, -0.70525, -0.66248,  0.04067])

    # Lower bound
    pnum = 'lower'
    shape_return(t_lower,pnum)

    # Upper bound
    pnum = 'upper'
    shape_return(t_upper,pnum)

    # Comparison
    pnum= 'Comparison_DNN'
    shape_return_comparison(t_truth,t_dnn,pnum)

    # Comparison
    pnum= 'Comparison_RL'
    shape_return_comparison(t_truth,t_rl,pnum)

