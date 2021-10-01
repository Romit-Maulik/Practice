# import os, yaml, sys, shutil
# current_dir = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(current_dir)

# Load YAML file for configuration - unique for each rank
config_file = open('config_'+str(rank)+'.yaml')
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
shutil.copyfile('config.yaml',data_paths['save_path']+'config.yaml')

if __name__ == '__main__':

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

    end_time = time()

    print('Total time taken:',end_time-start_time,' seconds')