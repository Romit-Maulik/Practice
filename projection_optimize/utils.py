import matplotlib.pyplot as plt
import numpy as np
np.random.seed(10)
import glob

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(dir_path+'/Shapes/'):
    os.mkdir(dir_path+'/Shapes')
else:
    files = glob.glob(dir_path+'/Shapes/*.png')
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

    if pnum == 166:       
        plt.figure()
        plt.plot(xx,cc,label='Optimized!')
        plt.ylim((-0.1,0.1))
    else:
        plt.figure()
        plt.plot(xx,cc,label='Iteration '+str(pnum))
        plt.ylim((-0.1,0.1))

    from constraints import t_base   
    t_base = np.asarray(t_base)

    xx, cc = plot_arrays(t_base,x0,x1,p)
    
    plt.plot(xx,cc,label='Base')
    plt.legend()
    plt.savefig('Shapes/Shape_'+"{0:0>2}".format(pnum)+'.png')
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

def plot_stats():
    import matplotlib
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    
    data = np.loadtxt('training_stats.csv',delimiter=',',skiprows=1)

    # Plot the losses and R2s
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
    ax1.semilogy(data[:,0],label='Training')
    ax1.semilogy(data[:,1],label='Validation')
    ax1.set_xlabel('Epochs',fontsize=16)
    ax1.set_ylabel('Loss',fontsize=16)
    ax1.legend(fontsize=16)
    ax1.tick_params(labelsize=12)

    ax2.plot(data[:,2],label='Training')
    ax2.plot(data[:,3],label='Validation')
    ax2.set_xlabel('Epochs',fontsize=16)
    ax2.set_ylabel('Coefficient of determination',fontsize=16)
    ax2.legend(fontsize=16)
    ax2.tick_params(labelsize=12)
    ax2.set_ylim((0.0,1.1))
    
    plt.tight_layout()
    plt.savefig('DNN_Training.pdf')
    plt.close()


def plot_scatter(true,predicted,fname):
    plt.figure()
    plt.plot(true,true,label='True')
    plt.scatter(true,predicted,s=3,label='Predicted',color='red')
    plt.xlabel('True '+fname,fontsize=16)
    plt.ylabel('Predicted '+fname,fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(labelsize=12)
    plt.savefig(fname+'_scatter.pdf')
    plt.close()


if __name__ == '__main__':

    print('Checking plotting of airfoil')
    t_base = np.asarray([0.1268, 0.467, 0.5834, 0.2103, -0.1268, -0.5425, -0.5096, 0.0581])
    t_lower = np.asarray([-0.08876, -0.3269, -0.40838, -0.14721, 0.1648, 0.70525, 0.66248, -0.04067])
    t_upper = np.asarray([0.1648, 0.6071, 0.75842, 0.27339, -0.08876, -0.37975, -0.35672, 0.07553])
    t_truth = np.asarray([0.1009, 0.3306, 0.6281, 0.1494, -0.1627, -0.6344, -0.5927, 0.0421])
    t_dnn = np.asarray([0.08876, 0.34271419, 0.43059023, 0.14721, -0.1648, -0.62865158, -0.55052489, 0.04578268]) # 30-30 
    # t_dnn = np.asarray([0.08876, 0.3496284, 0.40838,  0.14721, -0.1648, -0.68837999, -0.57474939,  0.04067]) # 15-15
    t_bo = np.asarray([0.1016,  0.3280,  0.4131,  0.1701, -0.1602, -0.6980, -0.4652,  0.0468])

    # DNN + BO
    t_dnn_bo = np.asarray([ 0.0957,  0.3493,  0.6934,  0.1560, -0.1389, -0.5759, -0.6443,  0.0411])

    # DNN lift constrained (lift of 0.9)
    # t_dnn_lift = np.asarray([0.08876, 0.34957082, 0.52368963, 0.14721, -0.1646667, -0.55007067, -0.50458895, 0.07454936])

    # DNN lift constrained (lift of 1.0)
    # t_dnn_lift = np.asarray([0.08876, 0.3269, 0.75842, 0.20406465, -0.1648, -0.64436213, -0.56695343, 0.05696154]) # 30-30
    t_dnn_lift = np.asarray([0.08876, 0.3269, 0.52007714, 0.27339, -0.1648, -0.70525, -0.61496747,  0.07553]) #15-15

    # Lower bound
    pnum = 'lower'
    shape_return(t_lower,pnum)

    # Upper bound
    pnum = 'upper'
    shape_return(t_upper,pnum)

    # Comparison
    pnum= 'Comparison: DNN'
    shape_return_comparison(t_truth,t_dnn,pnum)

    # Comparison
    pnum= 'Comparison: BO'
    shape_return_comparison(t_truth,t_bo,pnum)

    # Comparison
    pnum= 'Comparison: DNN + BO'
    shape_return_comparison(t_truth,t_bo,pnum)

    # Comparison
    pnum= 'Comparison: DNN Lift constrained'
    shape_return_comparison(t_truth,t_dnn_lift,pnum)

