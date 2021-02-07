import numpy as np
import matplotlib.pyplot as plt
from Parameters import K

def plot_coefficients(Ytilde,Ftilde):
    # Plot the true evolution of the coefficients for field and nonlinear term
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(Ytilde[0,:])
    ax[0].plot(Ytilde[1,:])
    ax[0].plot(Ytilde[2,:])
    ax[0].set_title('POD coefficients')
    
    ax[1].plot(Ftilde[0,:])
    ax[1].plot(Ftilde[1,:])
    ax[1].plot(Ftilde[2,:])
    ax[1].set_title('POD-DEIM coefficients')
    plt.show()

def plot_gp(Ytilde,label1,Ftilde,label2):
    # Plot GP rom predictions
    fig, ax = plt.subplots(ncols=3,figsize=(12,4))
    ax[0].plot(Ytilde[0,:],linewidth=2.5,label=label1)
    ax[0].plot(Ftilde[0,:],linewidth=2.5,label=label2)
    ax[0].set_title('Mode 1',fontsize=18)
    ax[0].grid()
    ax[0].set_xlabel('t',fontsize=18)
    ax[0].set_ylabel('$a_0$',fontsize=24)
    ax[0].tick_params(axis='both', which='major', labelsize=16)
    
    ax[1].plot(Ytilde[1,:],linewidth=2.5,label=label1)
    ax[1].plot(Ftilde[1,:],linewidth=2.5,label=label2)
    ax[1].set_title('Mode 2',fontsize=18)
    ax[1].grid()
    ax[1].set_xlabel('t',fontsize=18)
    ax[1].set_ylabel('$a_1$',fontsize=24)
    ax[1].tick_params(axis='both', which='major', labelsize=16)

    ax[2].plot(Ytilde[2,:],linewidth=2.5,label=label1)
    ax[2].plot(Ftilde[2,:],linewidth=2.5,label=label2)
    ax[2].set_title('Mode 3',fontsize=18)
    ax[2].grid()
    ax[2].set_xlabel('t',fontsize=18)
    ax[2].set_ylabel('$a_2$',fontsize=24)
    ax[2].tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout()
    plt.legend(fontsize=14,loc='upper right')
    plt.subplots_adjust(left=0.1)
    plt.show()

def plot_comparison(Ytilde_GP,Ytilde_DEIM,Ytilde_ML,Ytilde):
    fig, ax = plt.subplots(ncols=3,figsize=(18,6))
    ax[0].plot(Ytilde_GP[0,:],label='POD-GP')
    ax[0].plot(Ytilde_DEIM[0,:],label='POD-DEIM')
    ax[0].plot(Ytilde_ML[0,:],label='POD-ML')
    ax[0].plot(Ytilde[0,:],label='True')
    ax[0].set_title('Mode 1')
    
    ax[1].plot(Ytilde_GP[1,:],label='POD-GP')
    ax[1].plot(Ytilde_DEIM[1,:],label='POD-DEIM')
    ax[1].plot(Ytilde_ML[1,:],label='POD-ML')
    ax[1].plot(Ytilde[1,:],label='True')
    ax[1].set_title('Mode 2')

    ax[2].plot(Ytilde_GP[2,:],label='POD-GP')
    ax[2].plot(Ytilde_DEIM[2,:],label='POD-DEIM')
    ax[2].plot(Ytilde_ML[2,:],label='POD-ML')
    ax[2].plot(Ytilde[2,:],label='True')
    ax[2].set_title('Mode 3')

    plt.legend()
    plt.tight_layout()
    plt.show()

    print('Error calculation')
    print('POD GP:',error_calc(Ytilde,Ytilde_GP))
    print('POD DEIM:',error_calc(Ytilde,Ytilde_DEIM))
    print('POD ML:',error_calc(Ytilde,Ytilde_ML))

def error_calc(Ytilde,Ytilde_approx):
    return np.sqrt(np.sum((Ytilde[0,:]-Ytilde_approx[0,:])**2))


if __name__ == "__main__":
    print('Plotting functions file')