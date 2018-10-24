import numpy as np
import matplotlib.pyplot as plt
import itertools

def recursive_elm():
    #Set seed for repeatibility
    np.random.seed(10)

    #Data and model constants
    num_total_features = 11   #These are the number of features to be ranked
    num_outputs = 2           #Number of outputs
    num_hidden_neurons = 20   #Number of neurons in ELM

    num_samples = 2000        #Number of samples in training
    num_val_samples = 1000     #Number of samples in validation

    acceptance_ratio = 0.1    #This fraction of solutions will be accepted for rankings

    #Generate total data
    input_data = np.random.normal(size=(num_samples+num_val_samples,num_total_features))

    output_data = np.zeros(shape=(num_samples+num_val_samples, num_outputs), dtype='double')
    output_data[:, 0] = np.sin(input_data[:,1]+input_data[:,2]+input_data[:,3])
    output_data[:, 1] = np.cos(input_data[:,1]*input_data[:,3])*np.sin(input_data[:,3])

    #Training inputs
    training_inputs = input_data[0:num_samples,:]
    training_outputs = output_data[0:num_samples,:]

    #Training inputs
    validation_inputs = input_data[num_samples:,:]
    validation_outputs = output_data[num_samples:,:]

    #Feature list
    counter = np.zeros(shape=(num_total_features,),dtype='int')
    feature_range = np.arange(start=0,stop=num_total_features,dtype='int')

    #print(list(itertools.combinations(feature_range,2)))

    #Making a tracker of the total possible combinations of inputs
    combination_list = []
    for combination in range(1,len(feature_range)+1):#Atleast one feature must be used
        for subset in itertools.combinations(feature_range, combination):
            combination_list.append(np.asarray(subset))

    num_retain = int(acceptance_ratio*len(combination_list))

    #These combinations can be used as masks for the total input sampling (i.e. the columns)
    # print(training_inputs[:,combination_list[7]])
    # print(np.shape(training_inputs[:, combination_list[7]]))


    error_list = []
    for choice in range(len(combination_list)):
        choice_inputs = training_inputs[:,combination_list[choice]]
        #Set layer 1 weights
        num_inputs = np.shape(choice_inputs)[1]
        w1 = np.random.randn(num_inputs,num_hidden_neurons)
        b1 = np.random.randn(1,num_hidden_neurons)

        #multiply to get linear transform
        a1 = np.matmul(choice_inputs,w1)
        hidden_range = np.arange(0,num_hidden_neurons,dtype='int')
        a1[:,hidden_range] = a1[:,hidden_range] + b1[0,hidden_range]

        #Activate with tan sigmoid
        a1 = np.tanh(a1)

        #Use ELM (i.e., pseudoinverse projection to obtain w2)
        w2_opt = np.matmul(np.linalg.pinv(a1),training_outputs)

        #Find validation MSE
        validation_choice = validation_inputs[:,combination_list[choice]]
        preds = np.matmul(validation_choice,w1)
        preds[:,hidden_range] = preds[:,hidden_range] + b1[0,hidden_range]
        preds = np.tanh(preds)
        preds = np.matmul(preds,w2_opt)

        error = np.sum((preds - validation_outputs)**2)
        error_list.append(error)

    indices = np.array(error_list).argsort()
    indices = indices[0:num_retain]
    accepted_combinations = np.asarray(combination_list)[indices]

    for combination in range(0,len(accepted_combinations)):
        for count_val in range(num_total_features):
            if count_val in accepted_combinations[combination]:
                counter[count_val] = counter[count_val] + 1

    print(counter)
    y_pos = np.arange(len(counter))
    objects = []
    objects.append(str(feature_range)[:])

    plt.figure()
    plt.bar(y_pos, counter, align='center', alpha=0.5)
    plt.xticks(y_pos, feature_range)
    plt.ylabel('Feature occurence')
    plt.xlabel('Feature labels')

    plt.show()




recursive_elm()
