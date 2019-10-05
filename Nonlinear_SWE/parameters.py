import numpy as np

# This file for global parameters
Nx = 64      # Discretization in x direction
Ny = 64      # Discretization in y direction
ft = 0.2     # Final time
Lx = 1.0     # Length of domain in x direction
Ly = 1.0     # Length of domain in y direction
rho = 1.0          # Density of fluid [kg/m^3)]
dt = 0.0001  # discrete timestep
dx = Lx/Nx
dy = Ly/Ny
grav = 9.8

if __name__ == '__main__':
    print('This is the parameter file')
    print(dx)
