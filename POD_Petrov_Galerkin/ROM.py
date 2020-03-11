import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter

# Reproducibility
np.random.seed(10)


# Import relevant modules
from Parameters import K,M
from Compression import field_compression, nonlinear_compression
from Problem import collect_snapshots_field, collect_snapshots_nonlinear
from Plotting import plot_coefficients, plot_gp, plot_comparison
from GP import gp_evolution, pgp_evolution

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# This is the ROM assessment
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Snapshot collection for field
    # Note that columns of a snapshot/state are time always and a state vector is a column vector
    Y, Y_mean, Ytot = collect_snapshots_field() 
    F, F_mean, Ftot = collect_snapshots_nonlinear()

    # Field compression
    V, Ytilde = field_compression(Ytot,K) # K is the number of retained POD bases for field (and also dimension of reduced space)
    U, Ftilde_exact, P = nonlinear_compression(V,Ftot,M) # M is the number of retained POD basis for nonlinear term

    # Initialize ROM class
    ytilde_init = Ytilde[:,0].reshape(np.shape(Ytilde)[0],1)
    gp_rom = gp_evolution(V,U,P,ytilde_init)

    # ROM assessments - DEIM
    gp_rom.pod_deim_evolve()
    Ytilde_pod_deim = np.copy(gp_rom.state_tracker)
    # Plot comparison of POD-DEIM and truth
    plot_gp(Ytilde,'True',Ytilde_pod_deim,'POD-DEIM')
    # Plot comparison of DEIM nonlinear term from exact solution and through Rk3
    Ftilde_deim = gp_rom.nl_state_tracker
    
    # ROM assessments - GP
    gp_rom.pod_gp_evolve()
    Ytilde_pod_gp = np.copy(gp_rom.state_tracker)
    # Plot comparison of POD-GP and truth
    plot_gp(Ytilde,'True',Ytilde_pod_gp,'POD-GP')

    # ROM assessments - PGP
    pgp_rom = pgp_evolution(V,U,P,ytilde_init)
    pgp_rom.pod_pgp_evolve()
    Ytilde_pod_pgp = np.copy(pgp_rom.state_tracker)
    # Plot comparison of POD-PGP and truth
    plot_gp(Ytilde,'True',Ytilde_pod_pgp,'POD-PGP')


    # Plot comparison of all four techniques, ML,DEIM,GP and truth
    plot_comparison(Ytilde_pod_gp, Ytilde_pod_deim, Ytilde_pod_pgp, Ytilde) #Ytilde_GP,Ytilde_DEIM,Ytilde_ML,Ytilde

    print('Saving data')
    np.save('POD_True.npy',Ytilde)
    np.save('POD_GP.npy',Ytilde_pod_gp)
    np.save('POD_DEIM.npy',Ytilde_pod_deim)
    np.save('POD_PGP.npy',Ytilde_pod_pgp)
