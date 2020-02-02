import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter

# Reproducibility
np.random.seed(10)
tf.random.set_random_seed(10)

# Import relevant modules
from Parameters import K,M
from Compression import field_compression, nonlinear_compression
from Problem import collect_snapshots_field, collect_snapshots_nonlinear
from Plotting import plot_coefficients, plot_gp, plot_comparison
from ML import create_lstm_model, check_apriori_performance_lstm
from ML import create_slfn_model, check_apriori_performance_slfn
from GP import gp_evolution

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

    #Filter exact for stability
    Ftilde_filtered = np.copy(Ftilde_exact)
    for i in range(K):
        Ftilde_filtered[i,:] = savgol_filter(Ftilde_exact[i,:],65,2)
    # Plot comparison of filtered and exact nonlinear term modes
    plot_gp(Ftilde_exact,'Exact',Ftilde_filtered,'Filtered')

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

    # Do a simple fit for nonlinear term
    training_data = np.copy(Ftilde_filtered)
    mode='valid'
    trained_model_nl, _ = create_lstm_model(np.transpose(training_data),mode)

    # Assess performance in a-priori
    apriori_preds = check_apriori_performance_lstm(np.transpose(training_data),trained_model_nl)
    plot_gp(training_data,'DEIM coefficients',apriori_preds,'ML predictions')

    # Do aposteriori check
    gp_rom.lstm_gp_evolve(trained_model_nl)
    Ytilde_pod_ml = np.copy(gp_rom.state_tracker)
    # Plot comparison of POD-DEIM-ML and truth
    plot_gp(Ytilde,'True',Ytilde_pod_ml,'POD-ML')

    # Plot comparison of all four techniques, ML,DEIM,GP and truth
    plot_comparison(Ytilde_pod_gp, Ytilde_pod_deim, Ytilde_pod_ml, Ytilde) #Ytilde_GP,Ytilde_DEIM,Ytilde_ML,Ytilde

    print('Saving data')
    np.save('POD_True.npy',Ytilde)
    np.save('POD_GP.npy',Ytilde_pod_gp)
    np.save('POD_DEIM.npy',Ytilde_pod_deim)
    np.save('POD_ML.npy',Ytilde_pod_ml)
