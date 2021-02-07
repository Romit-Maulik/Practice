import numpy as np
import matplotlib.pyplot as plt

def plot_relative_difference(Ytilde,Ftilde):
    # Plot GP rom predictions
    fig, ax = plt.subplots(ncols=3,figsize=(12,4))
    ax[0].plot(np.abs(Ytilde[0,:]-Ftilde[0,:]),linewidth=2.5)
    ax[0].set_title('Mode 1',fontsize=18)
    ax[0].grid()
    ax[0].set_xlabel('t',fontsize=18)
    ax[0].set_ylabel('$a_0$',fontsize=24)
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    
    ax[1].plot(np.abs(Ytilde[1,:]-Ftilde[1,:]),linewidth=2.5)
    ax[1].set_title('Mode 2',fontsize=18)
    ax[1].grid()
    ax[1].set_xlabel('t',fontsize=18)
    ax[1].set_ylabel('$a_1$',fontsize=24)
    ax[1].tick_params(axis='both', which='major', labelsize=16)

    ax[2].plot(np.abs(Ytilde[2,:]-Ftilde[2,:]),linewidth=2.5)
    ax[2].set_title('Mode 3',fontsize=18)
    ax[2].grid()
    ax[2].set_xlabel('t',fontsize=18)
    ax[2].set_ylabel('$a_2$',fontsize=24)
    ax[2].tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    # plt.legend(fontsize=14)
    plt.subplots_adjust(left=0.1)
    
    plt.show()

Ytilde = np.load('POD_True.npy')
Ytilde_pod_gp = np.load('POD_GP.npy')
Ytilde_pod_deim = np.load('POD_DEIM.npy')
Ytilde_pod_ml = np.load('POD_ML.npy')

plot_relative_difference(Ytilde_pod_deim,Ytilde_pod_ml)