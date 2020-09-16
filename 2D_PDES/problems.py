# This initializes the problem class for SWE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from parameters import Nx, Ny

# Common functions for spatial discretizations
def state_reconstruction(q,Nx,Ny):
    # Weno5
    pad = 3
    qtemp = periodic_bc(q,pad)

    # Smoothness indicators in x
    beta_0 = 13.0/12.0*(qtemp[pad-2:pad+Nx-2,:]-2.0*qtemp[pad-1:pad+Nx-1,:]+qtemp[pad:Nx+pad,:])**2 \
            + 1.0/4.0*(qtemp[pad-2:pad+Nx-2,:]-4.0*qtemp[pad-1:pad+Nx-1,:]+3.0*qtemp[pad:Nx+pad,:])**2
    
    beta_1 = 13.0/12.0*(qtemp[pad-1:pad+Nx-1,:]-2.0*qtemp[pad:pad+Nx,:]+qtemp[pad+1:Nx+pad+1,:])**2 \
            + 1.0/4.0*(qtemp[pad-1:pad+Nx-1,:]-qtemp[pad+1:pad+Nx+1,:])**2
    
    beta_2 = 13.0/12.0*(qtemp[pad:pad+Nx,:]-2.0*qtemp[pad+1:pad+Nx+1,:]+qtemp[pad+2:Nx+pad+2,:])**2 \
            + 1.0/4.0*(3.0*qtemp[pad:pad+Nx,:]-4.0*qtemp[pad+1:pad+Nx+1,:]+qtemp[pad+2:Nx+pad+2,:])**2

    # nonlinear weights in x
    alpha_0 = (1.0/10.0)/((beta_0+1.0e-6)**2)
    alpha_1 = (6.0/10.0)/((beta_1+1.0e-6)**2)
    alpha_2 = (3.0/10.0)/((beta_2+1.0e-6)**2)

    # Find nonlinear weights
    w_0 = (alpha_0/(alpha_0+alpha_1+alpha_2))/6.0
    w_1 = (alpha_1/(alpha_0+alpha_1+alpha_2))/6.0
    w_2 = (alpha_2/(alpha_0+alpha_1+alpha_2))/6.0

    # Find state reconstructions in x - wave to right (at i+1/2)
    qxright = w_0*(2.0*qtemp[pad-2:pad+Nx-2,:]-7.0*qtemp[pad-1:pad+Nx-1,:]+11.0*qtemp[pad:pad+Nx,:]) \
          + w_1*(-qtemp[pad-1:pad+Nx-1,:]+5.0*qtemp[pad:pad+Nx,:]+2.0*qtemp[pad+1:pad+Nx+1,:]) \
          + w_2*(2.0*qtemp[pad:pad+Nx,:]+5.0*qtemp[pad+1:pad+Nx+1,:]-qtemp[pad+2:pad+Nx+2,:])

    # Find state reconstructions in x - wave to left (at i+1/2)
    qxleft = w_0*(2.0*qtemp[pad+2:pad+Nx+2,:]-7.0*qtemp[pad+1:pad+Nx+1,:]+11.0*qtemp[pad:pad+Nx,:]) \
          + w_1*(-qtemp[pad+1:pad+Nx+1,:]+5.0*qtemp[pad:pad+Nx,:]+2.0*qtemp[pad-1:pad+Nx-1,:]) \
          + w_2*(2.0*qtemp[pad:pad+Nx,:]+5.0*qtemp[pad-1:pad+Nx-1,:]-qtemp[pad-2:pad+Nx-2,:])

    qxleft = qxleft[:,pad:pad+Ny]
    qxright = qxright[:,pad:pad+Ny]

    # Smoothness indicators in y
    beta_0 = 13.0/12.0*(qtemp[:,pad-2:pad+Ny-2]-2.0*qtemp[:,pad-1:pad+Ny-1]+qtemp[:,pad:Ny+pad])**2 \
            + 1.0/4.0*(qtemp[:,pad-2:pad+Ny-2]-4.0*qtemp[:,pad-1:pad+Ny-1]+3.0*qtemp[:,pad:Ny+pad])**2
    
    beta_1 = 13.0/12.0*(qtemp[:,pad-1:pad+Ny-1]-2.0*qtemp[:,pad:pad+Ny]+qtemp[:,pad+1:Ny+pad+1])**2 \
            + 1.0/4.0*(qtemp[:,pad-1:pad+Ny-1]-qtemp[:,pad+1:pad+Ny+1])**2
    
    beta_2 = 13.0/12.0*(qtemp[:,pad:pad+Ny]-2.0*qtemp[:,pad+1:pad+Ny+1]+qtemp[:,pad+2:Ny+pad+2])**2 \
            + 1.0/4.0*(3.0*qtemp[:,pad:pad+Ny]-4.0*qtemp[:,pad+1:pad+Ny+1]+qtemp[:,pad+2:Ny+pad+2])**2

    # nonlinear weights in x
    alpha_0 = (1.0/10.0)/((beta_0+1.0e-6)**2)
    alpha_1 = (6.0/10.0)/((beta_1+1.0e-6)**2)
    alpha_2 = (3.0/10.0)/((beta_2+1.0e-6)**2)

    # Find nonlinear weights
    w_0 = (alpha_0/(alpha_0+alpha_1+alpha_2))/6.0
    w_1 = (alpha_1/(alpha_0+alpha_1+alpha_2))/6.0
    w_2 = (alpha_2/(alpha_0+alpha_1+alpha_2))/6.0

    # Find state reconstructions in y - qright (at i+1/2)
    qyright = w_0*(2.0*qtemp[:,pad-2:pad+Ny-2]-7.0*qtemp[:,pad-1:pad+Ny-1]+11.0*qtemp[:,pad:pad+Ny]) \
          + w_1*(-qtemp[:,pad-1:pad+Ny-1]+5.0*qtemp[:,pad:pad+Ny]+2.0*qtemp[:,pad+1:pad+Ny+1]) \
          + w_2*(2.0*qtemp[:,pad:pad+Ny]+5.0*qtemp[:,pad+1:pad+Ny+1]-qtemp[:,pad+2:pad+Ny+2])

    # Find state reconstructions in y - wave to left (at i+1/2)
    qyleft = w_0*(2.0*qtemp[:,pad+2:pad+Ny+2]-7.0*qtemp[:,pad+1:pad+Ny+1]+11.0*qtemp[:,pad:pad+Ny]) \
          + w_1*(-qtemp[:,pad+1:pad+Ny+1]+5.0*qtemp[:,pad:pad+Ny]+2.0*qtemp[:,pad-1:pad+Ny-1]) \
          + w_2*(2.0*qtemp[:,pad:pad+Ny]+5.0*qtemp[:,pad-1:pad+Ny-1]-qtemp[:,pad-2:pad+Ny-2])

    qyleft = qyleft[pad:pad+Nx,:]
    qyright = qyright[pad:pad+Nx,:]

    return qxleft, qxright, qyleft, qyright

def reimann_solve(spec_rad,fl,fr,ql,qr,dim):
    # Rusanov reimann solver
    pad = 3
    srt = periodic_bc(spec_rad,pad)
    if dim == 'x':
        srt = np.maximum.reduce([srt[pad-3:Nx+pad-3,pad:Ny+pad],srt[pad-2:Nx+pad-2,pad:Ny+pad],srt[pad-1:Nx+pad-1,pad:Ny+pad],\
            srt[pad:Nx+pad,pad:Ny+pad],srt[pad+1:Nx+pad+1,pad:Ny+pad],srt[pad+2:Nx+pad+2,pad:Ny+pad],srt[pad+3:Nx+pad+3,pad:Ny+pad]])
        flux = 0.5*(fr+fl) + 0.5*srt*(qr+ql)
        return flux
    else:
        srt = np.maximum.reduce([srt[pad:Nx+pad,pad-3:Ny+pad-3],srt[pad:Nx+pad,pad-2:Ny+pad-2],srt[pad:Nx+pad,pad-1:Ny+pad-1],\
            srt[pad:Nx+pad,pad:Ny+pad],srt[pad:Nx+pad,pad+1:Ny+pad+1],srt[pad:Nx+pad,pad+2:Ny+pad+2],srt[pad:Nx+pad,pad+3:Ny+pad+3]])
        flux = 0.5*(fr+fl) + 0.5*srt*(qr+ql)
        return flux

def periodic_bc(q,pad):
    qtemp = np.zeros(shape=(q.shape[0]+2*pad,q.shape[1]+2*pad),dtype='double')
    # Periodicity updates
    qtemp[pad:Nx+pad,pad:Ny+pad] = q[:,:]
    # x direction periodicity
    qtemp[0:pad,:] = qtemp[Nx-pad:Nx,:]
    qtemp[Nx+pad:,:] = qtemp[pad:2*pad,:]
    # y direction periodicity
    qtemp[:,0:pad] = qtemp[:,Ny-pad:Ny]
    qtemp[:,Ny+pad:] = qtemp[:,pad:2*pad]

    return qtemp

# Shallow water equations class
class shallow_water(object):
    """docstring for ClassName"""
    def __init__(self):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = 1.0
        self.Ly = 1.0

        self.rho = 1.0          # Density of fluid [kg/m^3)]
        self.dt = 0.0001  # discrete timestep
        self.dx = self.Lx/self.Nx
        self.dy = self.Ly/self.Ny
        self.grav = 9.8
        self.ft = 0.2


        x = np.linspace(-self.Lx/2, self.Lx/2, self.Nx)  # Array with x-points
        y = np.linspace(-self.Ly/2, self.Ly/2, self.Ny)  # Array with y-points

        # Meshgrid for plotting
        self.X, self.Y = np.meshgrid(x, y) 

        # Initialize fields
        self.initialize() 

        # Field storage for viz
        self.q1_list = []

        # Plot interval
        self.plot_interval = 20  

    def initialize(self):
        # There are three conserved quantities - initialize
        self.q1 = 10.0+(self.rho*np.exp(-((self.X-self.Lx/2.7)**2/(2*(0.05)**2) + (self.Y-self.Ly/4)**2/(2*(0.05)**2))))
        self.q2 = np.zeros(shape=(self.Nx,self.Ny),dtype='double')
        self.q3 = np.zeros(shape=(self.Nx,self.Ny),dtype='double')
        
    def flux_reconstruction(self,q1,q2,q3):
        spec_rad_x = self.spectral_radius(q1,q2)
        spec_rad_y = self.spectral_radius(q1,q3)

        q1xleft, q1xright, q1yleft, q1yright = state_reconstruction(q1,self.Nx,self.Ny)
        q2xleft, q2xright, q2yleft, q2yright = state_reconstruction(q2,self.Nx,self.Ny)
        q3xleft, q3xright, q3yleft, q3yright = state_reconstruction(q3,self.Nx,self.Ny)

        # Reconstructing fluxes for q1
        f1xleft = np.copy(q2xleft)
        f1xright = np.copy(q2xright)
        f1x = reimann_solve(spec_rad_x,f1xleft,f1xright,q1xleft,q1xright,'x')

        f1yleft = np.copy(q3yleft)
        f1yright = np.copy(q3yright)
        f1y = reimann_solve(spec_rad_y,f1yleft,f1yright,q1yleft,q1yright,'y')

        # Reconstructing fluxes for q2
        f2xleft = (q2xleft**2)/(q1xleft) + 0.5*(q1xleft**2)*(self.grav/self.rho)
        f2xright = (q2xright**2)/(q1xright) + 0.5*(q1xright**2)*(self.grav/self.rho)
        f2x = reimann_solve(spec_rad_x,f1xleft,f2xright,q2xleft,q2xright,'x')

        f2yleft = (q2yleft*q3yleft/q1yleft)
        f2yright = (q2yright*q3yright/q1yright)
        f2y = reimann_solve(spec_rad_y,f2yleft,f2yright,q2yleft,q2yright,'y')

        # Reconstructing fluxes for q3
        f3xleft = (q2xleft*q3xleft/q1xleft)
        f3xright = (q2xright*q3xright/q1xright)
        f3x = reimann_solve(spec_rad_x,f3xleft,f3xright,q3xleft,q3xright,'x')

        f3yleft = (q3yleft**2)/(q1yleft) + 0.5*(q1yleft**2)*(self.grav/self.rho)
        f3yright = (q3yright**2)/(q1yright) + 0.5*(q1yright**2)*(self.grav/self.rho)
        f3y = reimann_solve(spec_rad_y,f3yleft,f3yright,q3yleft,q3yright,'y')

        return f1x, f1y, f2x, f2y, f3x, f3y

    def spectral_radius(self,q1,q2):
        sound_speed = 2.0*np.sqrt(q1/self.rho*self.grav)
        u = q2/q1
        return np.maximum.reduce([np.abs(u+sound_speed),np.abs(u-sound_speed),\
                           np.abs(sound_speed)])

    def right_hand_side(self,q1,q2,q3):
        f1x, f1y, f2x, f2y, f3x, f3y = self.flux_reconstruction(q1,q2,q3) # these are all i+1/2

        # Periodicity
        pad = 1
        f1xtemp = periodic_bc(f1x,pad)
        f1ytemp = periodic_bc(f1y,pad)
        f2xtemp = periodic_bc(f2x,pad)
        f2ytemp = periodic_bc(f2y,pad)
        f3xtemp = periodic_bc(f3x,pad)
        f3ytemp = periodic_bc(f3y,pad)

        r1 = 1.0/self.dx*(f1xtemp[pad:Nx+pad,pad:Ny+pad]-f1xtemp[pad-1:Nx+pad-1,pad:Ny+pad]) + 1.0/self.dy*(f1ytemp[pad:Nx+pad,pad:Ny+pad]-f1ytemp[pad:Nx+pad,pad-1:Ny+pad-1])
        r2 = 1.0/self.dx*(f2xtemp[pad:Nx+pad,pad:Ny+pad]-f2xtemp[pad-1:Nx+pad-1,pad:Ny+pad]) + 1.0/self.dy*(f2ytemp[pad:Nx+pad,pad:Ny+pad]-f2ytemp[pad:Nx+pad,pad-1:Ny+pad-1])
        r3 = 1.0/self.dx*(f3xtemp[pad:Nx+pad,pad:Ny+pad]-f3xtemp[pad-1:Nx+pad-1,pad:Ny+pad]) + 1.0/self.dy*(f3ytemp[pad:Nx+pad,pad:Ny+pad]-f3ytemp[pad:Nx+pad,pad-1:Ny+pad-1])

        return -r1, -r2, -r3

    def integrate(self):
        # TVD-Rk3 time integration
        q1temp = np.copy(self.q1)
        q2temp = np.copy(self.q2)
        q3temp = np.copy(self.q3)

        r1, r2, r3 = self.right_hand_side(q1temp,q2temp,q3temp) # Note switch in sign

        q1temp[:,:] = self.q1[:,:] + self.dt*(r1[:,:])
        q2temp[:,:] = self.q2[:,:] + self.dt*(r2[:,:])
        q3temp[:,:] = self.q3[:,:] + self.dt*(r3[:,:])
        
        r1, r2, r3 = self.right_hand_side(q1temp,q2temp,q3temp) # Note switch in sign

        q1temp[:,:] = 0.75*self.q1[:,:] + 0.25*q1temp[:,:] + 0.25*self.dt*r1[:,:]
        q2temp[:,:] = 0.75*self.q2[:,:] + 0.25*q2temp[:,:] + 0.25*self.dt*r2[:,:]
        q3temp[:,:] = 0.75*self.q3[:,:] + 0.25*q3temp[:,:] + 0.25*self.dt*r3[:,:]
        
        r1, r2, r3 = self.right_hand_side(q1temp,q2temp,q3temp) # Note switch in sign

        self.q1[:,:] = (1.0/3.0)*self.q1[:,:] + (2.0/3.0)*q1temp[:,:] + (2.0/3.0)*self.dt*r1[:,:]
        self.q2[:,:] = (1.0/3.0)*self.q2[:,:] + (2.0/3.0)*q2temp[:,:] + (2.0/3.0)*self.dt*r2[:,:]
        self.q3[:,:] = (1.0/3.0)*self.q3[:,:] + (2.0/3.0)*q3temp[:,:] + (2.0/3.0)*self.dt*r3[:,:]


    def solve(self):
        self.t = 0
        plot_iter = 0
        while self.t < self.ft:
            print('Time is:',self.t)
            self.t = self.t + self.dt            
            self.integrate()
            
            if plot_iter == self.plot_interval:
                self.plot_fields_debug(self.q1)
                self.q1_list.append(self.q1)
                plot_iter = 0

            plot_iter = plot_iter + 1
            
        print('Solution finished')
  
    def plot_fields(self):
        fig = plt.figure(figsize = (11, 7))
        ax = Axes3D(fig)
        surf = ax.plot_surface(self.X, self.Y, self.q1, rstride = 1, cstride = 1,
            cmap = plt.cm.jet, linewidth = 0, antialiased = True)

        ax.set_title('Visualization', fontname = "serif", fontsize = 17)
        ax.set_xlabel("x [m]", fontname = "serif", fontsize = 16)
        ax.set_ylabel("y [m]", fontname = "serif", fontsize = 16)
        ax.set_zlabel('q1', fontname = "serif", fontsize = 16)
        plt.show()

        fig = plt.figure(figsize = (11, 7))
        ax = Axes3D(fig)
        surf = ax.plot_surface(self.X, self.Y, self.q2, rstride = 1, cstride = 1,
            cmap = plt.cm.jet, linewidth = 0, antialiased = True)

        ax.set_title('Visualization', fontname = "serif", fontsize = 17)
        ax.set_xlabel("x [m]", fontname = "serif", fontsize = 16)
        ax.set_ylabel("y [m]", fontname = "serif", fontsize = 16)
        ax.set_zlabel('q2', fontname = "serif", fontsize = 16)
        plt.show()

        fig = plt.figure(figsize = (11, 7))
        ax = Axes3D(fig)
        surf = ax.plot_surface(self.X, self.Y, self.q3, rstride = 1, cstride = 1,
            cmap = plt.cm.jet, linewidth = 0, antialiased = True)

        ax.set_title('Visualization', fontname = "serif", fontsize = 17)
        ax.set_xlabel("x [m]", fontname = "serif", fontsize = 16)
        ax.set_ylabel("y [m]", fontname = "serif", fontsize = 16)
        ax.set_zlabel('q3', fontname = "serif", fontsize = 16)
        plt.show()


    def plot_fields_debug(self,q1):
        fig = plt.figure(figsize = (11, 7))
        ax = Axes3D(fig)
        surf = ax.plot_surface(self.X, self.Y, q1, rstride = 1, cstride = 1,
            cmap = plt.cm.jet, linewidth = 0, antialiased = True)

        ax.set_title('Visualization', fontname = "serif", fontsize = 17)
        ax.set_xlabel("x [m]", fontname = "serif", fontsize = 16)
        ax.set_ylabel("y [m]", fontname = "serif", fontsize = 16)
        ax.set_zlim((9.8,10.2))
        plt.savefig(str(self.t)+'.png')

# Shallow water equations class
class burgers(object):
    """docstring for Burgers solver"""
    def __init__(self):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = 1.0
        self.Ly = 1.0
        self.dt = 0.0001  # discrete timestep
        self.dx = self.Lx/self.Nx
        self.dy = self.Ly/self.Ny
        self.ft = 0.1

        x = np.linspace(0.0, self.Lx, self.Nx)  # Array with x-points
        y = np.linspace(0.0, self.Ly, self.Ny)  # Array with y-points

        # Meshgrid for plotting
        self.X, self.Y = np.meshgrid(x, y) 

        # Initialize fields
        self.initialize() 

        # Field storage for viz
        self.q1_list = []

        # Plot interval
        self.plot_interval = 50    

    def initialize(self):
        # There is only one dependent variable u(x,y,t)
        self.q1 = np.zeros(shape=(self.Nx,self.Ny),dtype='double')
        for i in range(self.Nx):
            for j in range(self.Ny):
                if self.X[i,j] > 0.1 and self.X[i,j] < 0.6 and self.Y[i,j] > 0.1 and self.Y[i,j] < 0.6:
                    self.q1[i,j] = 10.0
                else:
                    self.q1[i,j] = 0.1

       
    def flux_reconstruction(self,q1):
        q1xleft, q1xright, q1yleft, q1yright = state_reconstruction(q1,self.Nx,self.Ny)

        # Reconstructing fluxes for q1
        f1xleft = np.copy(0.5*q1xleft**2)
        f1xright = np.copy(0.5*q1xright**2)
        f1x = reimann_solve(q1,f1xleft,f1xright,q1xleft,q1xright,'x')

        f1yleft = np.copy(0.5*q1yleft**2)
        f1yright = np.copy(0.5*q1yright**2)
        f1y = reimann_solve(q1,f1yleft,f1yright,q1yleft,q1yright,'y')

        return f1x, f1y

    def right_hand_side(self,q1):
        f1x, f1y = self.flux_reconstruction(q1) # these are all i+1/2

        # Periodicity
        pad = 1
        f1xtemp = periodic_bc(f1x,pad)
        f1ytemp = periodic_bc(f1y,pad)

        r1 = 1.0/self.dx*(f1xtemp[pad:Nx+pad,pad:Ny+pad]-f1xtemp[pad-1:Nx+pad-1,pad:Ny+pad]) + 1.0/self.dy*(f1ytemp[pad:Nx+pad,pad:Ny+pad]-f1ytemp[pad:Nx+pad,pad-1:Ny+pad-1])

        return -r1

    def integrate(self):
        # TVD-Rk3 time integration
        q1temp = np.copy(self.q1)

        r1 = self.right_hand_side(q1temp) # Note switch in sign
        q1temp[:,:] = self.q1[:,:] + self.dt*(r1[:,:])
        
        r1 = self.right_hand_side(q1temp) # Note switch in sign

        q1temp[:,:] = 0.75*self.q1[:,:] + 0.25*q1temp[:,:] + 0.25*self.dt*r1[:,:]
        r1 = self.right_hand_side(q1temp) # Note switch in sign

        self.q1[:,:] = (1.0/3.0)*self.q1[:,:] + (2.0/3.0)*q1temp[:,:] + (2.0/3.0)*self.dt*r1[:,:]

    def solve(self):
        self.t = 0
        plot_iter = 0
        while self.t < self.ft:
            print('Time is:',self.t)
            self.t = self.t + self.dt            
            self.integrate()
            
            if plot_iter == self.plot_interval:
                self.plot_fields_debug(self.q1)
                self.q1_list.append(self.q1)
                plot_iter = 0

            plot_iter = plot_iter + 1
            
        print('Solution finished')
  
    def plot_fields(self):
        fig = plt.figure(figsize = (11, 7))
        ax = Axes3D(fig)
        surf = ax.plot_surface(self.X, self.Y, self.q1, rstride = 1, cstride = 1,
            cmap = plt.cm.jet, linewidth = 0, antialiased = True)

        ax.set_title('Visualization', fontname = "serif", fontsize = 17)
        ax.set_xlabel("x [m]", fontname = "serif", fontsize = 16)
        ax.set_ylabel("y [m]", fontname = "serif", fontsize = 16)
        ax.set_zlabel('q1', fontname = "serif", fontsize = 16)
        plt.show()

    def plot_fields_debug(self,q1):
        fig = plt.figure(figsize = (11, 7))
        ax = Axes3D(fig)
        surf = ax.plot_surface(self.X, self.Y, q1, rstride = 1, cstride = 1,
            cmap = plt.cm.jet, linewidth = 0, antialiased = True)

        ax.set_title('Visualization', fontname = "serif", fontsize = 17)
        ax.set_xlabel("x [m]", fontname = "serif", fontsize = 16)
        ax.set_ylabel("y [m]", fontname = "serif", fontsize = 16)
        ax.set_zlim((0.0,15.0))
        plt.savefig(str(self.t)+'.png')