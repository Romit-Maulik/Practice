import ray
# Import other libraries
import os, yaml, sys, shutil
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

ray.init(address="auto")

# Export the function on workers with ressource utilization
@ray.remote(num_cpus=1, num_gpus=1)
def model_run(config):
    
    # Load YAML file for configuration - unique for each rank
    config_file = open('./case_builder/config_'+str(config)+'.yaml')
    configuration = yaml.load(config_file,Loader=yaml.FullLoader)

    data_paths = configuration['data_paths']
    subregion_paths = data_paths['subregions']
    operation_mode = configuration['operation_mode']
    model_choice = operation_mode['model_choice']
    hyperparameters = configuration.get('hyperparameters')

    config_file.close()

    # Location for test results
    if not os.path.exists(data_paths['save_path']):
        os.makedirs(data_paths['save_path'])
    # Save the configuration file for reference
    shutil.copyfile('./case_builder/config_'+str(config)+'.yaml',data_paths['save_path']+'config.yaml')

    from time import time

    start_time = time()

    import numpy as np
    np.random.seed(10)
    from lstm_archs import emulator

    # Loading data
    num_modes = hyperparameters[6]
    train_data = np.load(data_paths['training_coefficients']).T[:,:num_modes]

    # Initialize model
    lstm_model = emulator(train_data,data_paths['save_path'],hyperparameters,model_choice)
    
    # Training model
    if operation_mode['train']:
        lstm_model.train_model()

    # Regular testing of model
    if operation_mode['test']:
        
        test_data = np.load(data_paths['testing_coefficients']).T[:,:num_modes]
        true, forecast = lstm_model.regular_inference(test_data)

        if not os.path.exists(data_paths['save_path']+'/Regular/'):
            os.makedirs(data_paths['save_path']+'/Regular/')    
        np.save(data_paths['save_path']+'/Regular/True.npy',true)
        np.save(data_paths['save_path']+'/Regular/Predicted.npy',forecast)

    # 3DVar testing of model
    if operation_mode['perform_var']:
        
        test_data = np.load(data_paths['testing_coefficients']).T[:,:num_modes]
        train_fields = np.load(data_paths['training_fields']).T
        test_fields = np.load(data_paths['da_testing_fields']).T
        pod_modes = np.load(data_paths['pod_modes'])[:,:num_modes]
        training_mean = np.load(data_paths['training_mean'])

        retval = lstm_model.variational_inference(test_data,train_fields,test_fields,pod_modes,training_mean)

        if not os.path.exists(data_paths['save_path']+'3DVar/'):
            os.makedirs(data_paths['save_path']+'3DVar/')
        np.save(data_paths['save_path']+'/3DVar/True.npy',retval[0])
        np.save(data_paths['save_path']+'/3DVar/Predicted.npy',retval[1])
        np.save(data_paths['save_path']+'/3DVar/Objective_Functions.npy',retval[2])

    # Constrained 3DVar testing of model
    if operation_mode['constrained_var']:

        test_data = np.load(data_paths['testing_coefficients']).T[:,:num_modes]
        train_fields = np.load(data_paths['training_fields']).T
        test_fields = np.load(data_paths['da_testing_fields']).T
        pod_modes = np.load(data_paths['pod_modes'])[:,:num_modes]
        training_mean = np.load(data_paths['training_mean'])

        num_fixed_modes = hyperparameters[7]
        true, forecast = lstm_model.constrained_variational_inference(test_data,train_fields,test_fields,pod_modes,training_mean,num_fixed_modes)

        if not os.path.exists(data_paths['save_path']+'3DVar_Constrained/'):
            os.makedirs(data_paths['save_path']+'3DVar_Constrained/')
        np.save(data_paths['save_path']+'/3DVar_Constrained/True.npy',true)
        np.save(data_paths['save_path']+'/3DVar_Constrained/Predicted.npy',forecast)
            

    if operation_mode['perform_analyses']:

        from post_analyses import perform_analyses, plot_obj

        num_inputs = hyperparameters[1]
        num_outputs = hyperparameters[2]
        var_time = hyperparameters[4]
        cadence = hyperparameters[8]
        output_gap = hyperparameters[10]

        if os.path.isfile(data_paths['save_path']+'/Regular/Predicted.npy'):
            forecast = np.load(data_paths['save_path']+'/Regular/Predicted.npy')
            test_fields = np.load(data_paths['da_testing_fields'])
            perform_analyses(data_paths,var_time,cadence,num_inputs,num_outputs,output_gap,num_modes,
                            test_fields,forecast,
                            data_paths['save_path']+'/Regular/',subregion_paths)
        else:
            print('No forecast for the test data. Skipping analyses.')
        
        if os.path.isfile(data_paths['save_path']+'/3DVar/Predicted.npy'):
            forecast = np.load(data_paths['save_path']+'/3DVar/Predicted.npy')
            test_fields = np.load(data_paths['da_testing_fields'])
            perform_analyses(data_paths,var_time,cadence,num_inputs,num_outputs,output_gap,num_modes,
                            test_fields,forecast,
                            data_paths['save_path']+'/3DVar/',subregion_paths)

            # obj_array = np.load(data_paths['save_path']+'/3DVar/Predicted.npy')
            # plot_obj(obj_array,data_paths['save_path']+'/3DVar/')

        else:
            print('No forecast for the test data with 3D Var. Skipping analyses.')


        if os.path.isfile(data_paths['save_path']+'/3DVar_Constrained/Predicted.npy'):
            forecast = np.load(data_paths['save_path']+'/3DVar_Constrained/Predicted.npy')
            test_fields = np.load(data_paths['da_testing_fields'])
            perform_analyses(data_paths,var_time,cadence,num_inputs,num_outputs,output_gap,num_modes,
                            test_fields,forecast,
                            data_paths['save_path']+'/3DVar_Constrained/',subregion_paths)
        else:
            print('No forecast for the test data with constrained 3D Var. Skipping analyses.')

    # Perform postprocessing
    import matplotlib.pyplot as plt
    save_path = data_paths['save_path']
    subregion_paths = data_paths['subregions']
    
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

    end_time = time()

    print('Total time taken for training and analysis:',end_time-start_time,' seconds from configuration ', config)

CONFIGS = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
ray.get([model_run.remote(config) for config in CONFIGS])