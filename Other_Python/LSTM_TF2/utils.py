import matplotlib.pyplot as plt
import numpy as np
np.random.seed(10)

# Coefficient of determination
def coeff_determination(y_pred, y_true): #Order of function inputs is important here
    y_pred = y_pred.numpy()
    num_points = y_true.shape[0]

    y_true = y_true.reshape(num_points,-1)
    y_pred = y_pred.reshape(num_points,-1)

    SS_res =  np.sum(np.square( y_true-y_pred )) 
    SS_tot = np.sum(np.square( y_true - np.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + 2.22044604925e-16) )