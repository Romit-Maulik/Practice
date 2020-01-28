import numpy as np

# Global parameters of WENO SWE
Nx = 64      # Discretization in x direction
Ny = 64      # Discretization in y direction
ft = 0.5     # Final time
Lx = 1.0     # Length of domain in x direction
Ly = 1.0     # Length of domain in y direction
rho = 1.0          # Density of fluid [kg/m^3)]
dt = 0.001  # discrete timestep
dx = Lx/Nx
dy = Ly/Ny
grav = 9.8

# Global parameters of SWE ROM
K = 4

# Mode of solver
fvm_solve = False
plot_viz = False
num_steps_per_plot = 1 # The number of timesteps per snapshot

if __name__ == '__main__':
    print('This is the parameter file')
    print(dx)
