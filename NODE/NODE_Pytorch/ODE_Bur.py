import sys
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad, functional
import scipy.io
import seaborn as sns
from matplotlib.colors import ListedColormap

###############################################################################
# Arguments from the submit file
###############################################################################
parser = argparse.ArgumentParser('ODE demo')
# These are the relevant sampling parameters
parser.add_argument('--data_size', type=int, default=100)  #IC from the simulation
parser.add_argument('--dt',type=float,default=0)
parser.add_argument('--batch_time', type=int, default=9)   #Samples a batch covers (this is 10 snaps in a row in data_size)
parser.add_argument('--batch_size', type=int, default=20)   #Number of IC to calc gradient with each iteration

parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--niters', type=int, default=60)       #Iterations of training
parser.add_argument('--test_freq', type=int, default=20)    #Frequency for outputting test loss
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--train', action='store_true')
args = parser.parse_args()

#Add 1 to batch_time to include the IC
args.batch_time+=1 

# Determines what solver to use
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    # This is the default
    from torchdiffeq import odeint

# Check if there are gpus
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# Define zero
t_zero = torch.FloatTensor([0])

###############################################################################
# Classes
###############################################################################

# This is the class that contains the NN that estimates the RHS
class ODEFunc_LES(nn.Module):
    def __init__(self,N):
        super(ODEFunc_LES, self).__init__()

        self.lin=nn.Sequential(nn.Linear(N, N,bias=False),)
        self.grad_m=nn.Sequential(nn.Linear(N, N,bias=False),)
        self.grad_p=nn.Sequential(nn.Linear(N, N,bias=False),)

        for m in self.lin.modules():
            if isinstance(m, nn.Linear):
                m.weight=nn.Parameter(torch.from_numpy(Linear(N)).float())
                m.weight.requires_grad=False

        for m in self.grad_m.modules():
            if isinstance(m, nn.Linear):
                m.weight=nn.Parameter(torch.from_numpy(grad_upwind(N)).float())
                m.weight.requires_grad=False

        for m in self.grad_p.modules():
            if isinstance(m, nn.Linear):
                m.weight=nn.Parameter(torch.from_numpy(-grad_upwind(N).T).float())
                m.weight.requires_grad=False
        

    def forward(self, t, y):
        return self.lin(y) - self.grad_m(y)*torch.max(y,t_zero.expand_as(y)) - self.grad_p(y)*torch.min(y,t_zero.expand_as(y))


# This is the class that contains the NN that estimates the RHS
class ODEFunc(nn.Module):
    def __init__(self,N):
        super(ODEFunc, self).__init__()
        # Change the NN architecture here
        self.net = nn.Sequential(
            nn.Linear(N, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200,200),
            nn.ReLU(),
            nn.Linear(200, N),
        )

        self.lin=nn.Sequential(nn.Linear(N, N,bias=False),)
        self.grad_m=nn.Sequential(nn.Linear(N, N,bias=False),)
        self.grad_p=nn.Sequential(nn.Linear(N, N,bias=False),)

        for m in self.lin.modules():
            if isinstance(m, nn.Linear):
                m.weight=nn.Parameter(torch.from_numpy(Linear(N)).float())
                m.weight.requires_grad=False

        for m in self.grad_m.modules():
            if isinstance(m, nn.Linear):
                m.weight=nn.Parameter(torch.from_numpy(grad_upwind(N)).float())
                m.weight.requires_grad=False

        for m in self.grad_p.modules():
            if isinstance(m, nn.Linear):
                m.weight=nn.Parameter(torch.from_numpy(-grad_upwind(N).T).float())
                m.weight.requires_grad=False

    def forward(self, t, y):
        return self.lin(y) - self.grad_m(y)*torch.max(y,t_zero.expand_as(y)) - self.grad_p(y)*torch.min(y,t_zero.expand_as(y)) + self.net(y)

    # The following is not used for now.
    def transport_constraint(self,t,y):

        closure = self.net(y)
        closure_linear = self.lin(closure)
        closure_nonlinear = self.grad_m(closure)*torch.max(closure,t_zero.expand_as(y)) + self.grad_p(closure)*torch.min(closure,t_zero.expand_as(y))

        # Compute temporal derivative using Euler
        dcdt = torch.zeros(closure.size()[0]-1,closure.size()[1],closure.size()[2])
        for i in range(1,int(closure.size()[0])):
            dt = t[i]-t[i-1]
            dcdt[i-1] = (closure[i]-closure[i-1])/(dt)
            
        closure_linear=closure_linear[1:]
        closure_nonlinear=closure_nonlinear[1:]

        constraint_error = torch.mean((dcdt + closure_nonlinear - closure_linear)**2)

        # This is the constraint on the closure term
        return constraint_error

# This class is used for updating the gradient
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

###############################################################################
# Functions for the RHS
###############################################################################
def Linear(N):
    alpha = 8.0e-4
    dx2=np.zeros((N,N))
    for i in range(N):
        
        dx2[i,i]=-2
        if i==0:
            dx2[i,-1]=1
            dx2[i,i+1]=1
        elif i==N-1:
            dx2[i,i-1]=1
            dx2[i,0]=1       
        else:
            dx2[i,i-1]=1
            dx2[i,i+1]=1
    
    dx=1/N
    A=alpha*(1/dx**2)*dx2

    return A


def grad_central(N):
    dx2=np.zeros((N,N))
    for i in range(N):
        
        dx2[i,i]=0
        if i==0:
            dx2[i,i+1]=1
            dx2[i,-1]=-1
        elif i==N-1:
            dx2[i,0]=1
            dx2[i,i-1]=-1       
        else:
            dx2[i,i+1]=1
            dx2[i,i-1]=-1
    
    dx=1/N
    A=(1/(2.0*dx))*dx2

    return A


def grad_upwind(N):
    dx2=np.zeros((N,N))
    for i in range(N):
        
        dx2[i,i]=3.0
        
        if i==0:
            dx2[i,-1]=-4.0
            dx2[i,-2]=1.0
        
        elif i==1:
            dx2[i,-1]=1.0
            dx2[i,i-1]=-4.0
        
        else:
            dx2[i,i-1]=-4.0       
            dx2[i,i-2]=1.0       
    
    dx=1/N
    A=(1/(2.0*dx))*dx2

    return A

# Gets a batch of y from the data evolved forward in time (default 20)
def get_batch(ttorch,utorch):
    [IC,length,_]=utorch.shape

    batch_size=args.batch_size
    batch_time=args.batch_time

    x=[[j,i] for i in range(length-batch_time) for j in range(IC)]
    lis=[x[i] for i in np.random.choice(len(x),batch_size,replace=False)]

    for i in range(len(lis)):
        if i==0:
            batch_y0=utorch[lis[i][0],lis[i][1]][None,:]
            batch_t = ttorch[lis[i][0],lis[i][1]:lis[i][1]+batch_time][None,:]-ttorch[lis[i][0],lis[i][1]][None,None]
            batch_y = torch.stack([utorch[lis[i][0],lis[i][1]+j] for j in range(batch_time)], dim=0)[:,None,:]
        else:
            batch_y0=torch.cat((batch_y0,utorch[lis[i][0],lis[i][1]][None,:]))
            batch_t=torch.cat((batch_t,ttorch[lis[i][0],lis[i][1]:lis[i][1]+batch_time][None,:]-ttorch[lis[i][0],lis[i][1]][None,None]))
            batch_y=torch.cat((batch_y,torch.stack([utorch[lis[i][0],lis[i][1]+j] for j in range(batch_time)], dim=0)[:,None,:]),axis=1)

    return batch_y0, batch_t, batch_y

# Plotting scripts
def plotting(true_y, pred_y,xlabel='x',ylabel='y',zlabel='z'):
    plt.figure(figsize=(7.5,6))
    ax=plt.subplot(projection='3d')
    #ax.set_xlim([-25,25])
    #ax.set_ylim([-25,25])
    #ax.set_zlim([0,40])
    plt.plot(true_y.detach().numpy()[:, 0, 0],true_y.detach().numpy()[:, 0, 1],true_y.detach().numpy()[:, 0, 2],'.',color='black',linewidth=.5,markersize=1,alpha=1)
    plt.plot(pred_y.detach().numpy()[:, 0, 0],pred_y.detach().numpy()[:, 0, 1],pred_y.detach().numpy()[:, 0, 2],'.',color='red',linewidth=.5,markersize=1,alpha=.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend(('True','Pred'))
    #ax.view_init(30, 30)

def plot2d(u_true,u_pred):
    N=u_true.shape[0]
    colors=sns.diverging_palette(240, 10, n=9,as_cmap=True)

    col=3
    row=3
    scale=1.2
    fig, axs = plt.subplots(col, row, sharex='col', sharey='row',figsize=(scale*col,scale*row))

    for i in range(col):
        for j in range(row):
            T=10*(i*col+j)
            axs[i][j].plot( np.linspace(0,1,N), u_true[:,T])
            axs[i][j].plot( np.linspace(0,1,N), u_pred[:,T],'--')
            #axs[i][j].legend((r'$u$'+str(T),r'$\tilde{u}$'+str(T)))

def plot_trajectories(t_true,u_true,t_pred,u_pred):
    N=u_true.shape[0]

    colors=sns.diverging_palette(240, 10, n=9,as_cmap=True)
    colors2=sns.diverging_palette(240, 10, n=41)
    colors2 = ListedColormap(colors2[20:])
    fig, axs = plt.subplots(3, 1, sharex='col', sharey='row',figsize=(4,4))
    (ax1),(ax2),(ax3) = axs
    im=ax1.pcolormesh(t_true, np.linspace(0,2*np.pi,N), u_true, shading='gouraud', cmap=colors,vmin=-1,vmax=1)
    im2=ax2.pcolormesh(t_pred, np.linspace(0,2*np.pi,N), u_pred, shading='gouraud', cmap=colors,vmin=-1,vmax=1)
    im3=ax3.pcolormesh(t_pred, np.linspace(0,2*np.pi,N), np.abs(u_pred-u_true), shading='gouraud', cmap=colors2,vmin=0,vmax=1)
    
    ax1.set(ylabel='x')
    ax2.set(ylabel='x')  
    ax3.set(ylabel='x')
    ax3.set(xlabel='t')
    
    cax = fig.add_axes([.91, 0.6575, 0.015, 0.225]) #left bottom width height
    cb=fig.colorbar(im, cax=cax, orientation='vertical')
    cb.set_label(r'$u$')
    cax2 = fig.add_axes([.91, 0.39, 0.015, 0.225]) #left bottom width height
    cb2=fig.colorbar(im2, cax=cax2, orientation='vertical')
    cb2.set_label(r'$u_{NN}$')
    cax3 = fig.add_axes([.91, 0.124, 0.015, 0.222]) #left bottom width height
    cb3=fig.colorbar(im3, cax=cax3, orientation='vertical')
    cb3.set_label(r'$|u-u_{NN}|$')
    
    for ax in axs.flat:
        ax.label_outer()

# For outputting info when running on compute nodes
def output(text):
    # Output epoch and Loss
    name='Out.txt'
    newfile=open(name,'a+')
    newfile.write(text)
    newfile.close()

if __name__ == '__main__':

    coarse_deg = 16

    if args.train:

        ###########################################################################
        # Import Data
        ###########################################################################
        [u,t]=pickle.load(open('./Data_T5_IC1000.p','rb'))
        u = np.asarray(u)

        # plt.figure()
        # plt.plot(np.arange(512)/512.0,u[0,10,:])
        # plt.plot(np.arange(512//coarse_deg)/(512.0/coarse_deg),u[0,10,::coarse_deg])
        # plt.show()

        u = u[:,:,::coarse_deg]

        utorch=torch.Tensor(u)
        utorch=utorch.type(torch.FloatTensor)
        [IC,T,N]=utorch.shape
        ttorch=torch.Tensor(t)
        ttorch=ttorch.type(torch.FloatTensor)


        ###########################################################################
        # Initialize NN for learning the RHS and setup optimization parms
        ###########################################################################
        func = ODEFunc(N)
        optimizer = optim.Adam(func.parameters(), lr=1e-3) #optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.niters/3), gamma=0.1)
        end = time.time()
        time_meter = RunningAverageMeter(0.97)
        loss_meter = RunningAverageMeter(0.97)

        err=[]
        test_err=[]
        ii = 0
        ###########################################################################
        # Optimization iterations
        ###########################################################################
        ex=0
        for itr in range(1, args.niters + 1):

            # Get the batch and initialzie the optimizer
            optimizer.zero_grad()
            batch_y0, batch_t, batch_y = get_batch(ttorch,utorch)

            batch_t=batch_t[0] # Ask Alec
            #batch_t=torch.swapaxes(batch_t,0,1)

            if itr==1:
                output('Batch Time Units: '+str(batch_t.detach().numpy()[-1])+'\n')

            # Make a prediction and calculate the loss
            pred_y = odeint(func, batch_y0, batch_t)

            
            loss = 1000.0*torch.mean(torch.abs(pred_y - batch_y)) # Compute the mean (because this includes the IC it is not as high as it should be)
            # loss = loss + func.transport_constraint(batch_t,pred_y) # Penalize a transport constraint for the closure term?
            
            loss.backward() #Computes the gradient of the loss (w.r.t to the parameters of the network?)
            # Use the optimizer to update the model
            optimizer.step()
            scheduler.step()
            
            # Print out the Loss and the time the computation took
            time_meter.update(time.time() - end)
            loss_meter.update(loss.item())
            if itr % args.test_freq == 0:
                with torch.no_grad():
                    # Testing loss
                    # batch_y0, batch_t, batch_y = get_batch(test_t,test_y)
                    # pred_y = odeint(func, batch_y0, batch_t)
                    # test_loss = torch.mean(torch.abs(pred_y - batch_y))

                    err.append(loss.item())
                    #test_err.append(test_loss.item())
                    #output('Iter {:04d} | Total Loss {:.6f} | Val Loss {:.6f} | Time {:.6f}'.format(itr, loss.item(),test_loss.item(),time.time() - end)+'\n')
                    output('Iter {:04d} | Total Loss {:.6f} | Time {:.6f}'.format(itr, loss.item(),time.time() - end)+'\n')
                    ii += 1
            end = time.time()
        
        ###########################################################################
        # Plot results and save the model
        ###########################################################################
        torch.save(func, 'model.pt')
        #pickle.dump(func,open('model.p','wb'))

        # Plot the learning
        plt.figure()
        plt.plot(np.arange(args.test_freq,args.niters+1,args.test_freq),np.asarray(err),'-')
        #plt.legend('Train')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.savefig('Error_v_Epochs.png')
        plt.close()

    else:

        [u,t]=pickle.load(open('./Data_T5_IC100.p','rb'))
        u = np.asarray(u)
        u = u[:,:,::coarse_deg]
        
        utorch=torch.Tensor(u)
        utorch=utorch.type(torch.FloatTensor)
        [IC,T,N]=utorch.shape
        ttorch=torch.Tensor(t)
        ttorch=ttorch.type(torch.FloatTensor)

        # LES testing
        func = ODEFunc_LES(N)
        for ex in range(4,5): # Change this for all ICs
            pred_y = odeint(func, utorch[ex,0,:], ttorch[ex,:])

            # plt.figure()
            # plt.plot(np.arange(512//coarse_deg)/(512.0/coarse_deg),u[ex][-1,:])
            # plt.plot(np.arange(512//coarse_deg)/(512.0/coarse_deg),pred_y.detach().numpy()[-1,:])
            # plt.show()
            # exit()

            plot_trajectories(t[ex],u[ex].transpose(),t[ex],pred_y.detach().numpy().transpose())
            plt.savefig('LES_Traj_'+str(ex)+'.png')

            plot2d(u[ex].transpose(),pred_y.detach().numpy().transpose())
            plt.savefig('LES_Traj2D_'+str(ex)+'.png')

        
        # ###########################################################################
        # # Load model and perform testing with NN Closure
        # ###########################################################################
        
        func = torch.load('model.pt')
        for ex in range(4,5): # Change this for all ICs
            pred_y = odeint(func, utorch[ex,0,:], ttorch[ex,:])

            plot_trajectories(t[ex],u[ex].transpose(),t[ex],pred_y.detach().numpy().transpose())
            plt.savefig('Traj_'+str(ex)+'.png')

            plot2d(u[ex].transpose(),pred_y.detach().numpy().transpose())
            plt.savefig('Traj2D_'+str(ex)+'.png')
