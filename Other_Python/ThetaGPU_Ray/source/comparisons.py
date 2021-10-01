import os, sys, yaml
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

import numpy as np
import matplotlib.pyplot as plt

# Load YAML file for configuration
config_file = open('config.yaml')
configuration = yaml.load(config_file,Loader=yaml.FullLoader)
data_paths = configuration['data_paths']
save_path = data_paths['save_path']
subregion_paths = data_paths['subregions']
config_file.close()

# Load standard results
if os.path.exists(data_paths['save_path']+'/Regular/'):
    persistence_maes = np.loadtxt(data_paths['save_path']+'/Regular/'+'persistence_maes.txt')
    climatology_maes = np.loadtxt(data_paths['save_path']+'/Regular/'+'climatology_maes.txt')
    regular_maes = np.loadtxt(data_paths['save_path']+'/Regular/'+'predicted_maes.txt')
else:
    print('Regular forecasts do not exist. Stopping.')
    exit()

var_data = True
if os.path.exists(data_paths['save_path']+'/3DVar/'):
    var_maes = np.loadtxt(data_paths['save_path']+'/3DVar/'+'predicted_maes.txt')
else:
    print('Warning: 3DVar forecasts do not exist.')
    var_data = False

cvar_data = True
if os.path.exists(data_paths['save_path']+'/3DVar_Constrained/'):
    cons_var_maes = np.loadtxt(data_paths['save_path']+'/3DVar_Constrained/'+'predicted_maes.txt')
else:
    print('Warning: Constrained 3DVar forecasts do not exist.')
    cvar_data = False

iter_num = 0
for subregion in subregion_paths:
    fname = subregion.split('/')[-1].split('_')[0]
    plt.figure()
    plt.title('MAE for '+fname)
    plt.plot(persistence_maes[:,iter_num],label='Persistence')
    plt.plot(climatology_maes[:,iter_num],label='Climatology')
    plt.plot(regular_maes[:,iter_num],label='Regular')

    if var_data:
        plt.plot(var_maes[:,iter_num],label='3DVar')

    if cvar_data:
        plt.plot(cons_var_maes[:,iter_num],label='Constrained 3DVar')
    
    plt.legend()
    plt.xlabel('Timesteps')
    plt.ylabel('MAE')
    plt.savefig(save_path+'/'+fname+'.png')
    plt.close()

    iter_num+=1


iter_num = -1
fname = 'everything'
plt.figure()
plt.title('MAE for '+fname)
plt.plot(persistence_maes[:,iter_num],label='Persistence')
plt.plot(climatology_maes[:,iter_num],label='Climatology')
plt.plot(regular_maes[:,iter_num],label='Regular')

if var_data:
    plt.plot(var_maes[:,iter_num],label='3DVar')

if cvar_data:
    plt.plot(cons_var_maes[:,iter_num],label='Constrained 3DVar')

plt.legend()
plt.xlabel('Timesteps')
plt.ylabel('MAE')
plt.savefig(save_path+'/'+fname+'.png')
plt.close()

