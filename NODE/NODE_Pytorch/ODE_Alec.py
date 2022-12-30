#! /home/linot/anaconda3/bin/python3
import sys
sys.path.insert(0, '/home/linot/odeNet/torchdiffeq') # Change to torchdiffeq path
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
import scipy.io

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

###############################################################################
# Classes
###############################################################################

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
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

        for m in self.lin.modules():
            if isinstance(m, nn.Linear):
                m.weight=nn.Parameter(torch.from_numpy(Linear(N)).float())
                m.weight.requires_grad=False

    def forward(self, t, y):
        # This is the evolution with the NN
        return self.lin(y)+self.net(y)

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
# Functions
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

# For outputting info when running on compute nodes
def output(text):
    # Output epoch and Loss
    name='Out.txt'
    newfile=open(name,'a+')
    newfile.write(text)
    newfile.close()

if __name__ == '__main__':

    ###########################################################################
    # Import Data
    ###########################################################################
    [u,t]=pickle.load(open('./Data_T5_IC1000.p','rb'))
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
        batch_t=batch_t[0]
        #batch_t=torch.swapaxes(batch_t,0,1)
        #print(batch_t.shape)
        if itr==1:
            output('Batch Time Units: '+str(batch_t.detach().numpy()[-1])+'\n')

        # Make a prediction and calculate the loss
        pred_y = odeint(func, batch_y0, batch_t)
        print(pred_y.shape)
        print(batch_y.shape)
        loss = torch.mean(torch.abs(pred_y - batch_y)) # Compute the mean (because this includes the IC it is not as high as it should be)
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
