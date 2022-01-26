import numpy as np
np.random.seed(10)

from lstm_archs import grid_lstm

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LSTM training of time-series data')
    parser.add_argument('--train', help='Do training', action='store_true') #Train a network or use a trained network for inference
    args = parser.parse_args()

    # Loading data train and test
    raw_data = np.load('SST_Train.npy').T
    train_data = [raw_data[:200],raw_data[:200],raw_data[:200]]
    raw_test_data = np.load('SST_Test.npy').T
    test_data = [raw_test_data[:200],raw_test_data[:200],raw_test_data[:200]]

    # Forecast parameters
    resolution_list = [1,1,1] # the data element shape divided by resolution should be constant

    # These are input variable indices
    input_idx = [0,1,2]
    # Output variable indices
    output_idx = [2]
    # Horizons of input and forecasts
    input_horizon = 8 # These are days
    output_horizon = 10 # These are days
    
    # Initialize model
    lstm_model = grid_lstm(train_data,input_idx,output_idx,input_horizon,output_horizon,resolution_list)
    # Training model
    if args.train:
        lstm_model.train_model(batch_size=20,num_epochs=100) # Train and exit
    else:
        data = np.load('SST_Test.npy').T
        true, predicted = lstm_model.model_inference(test_data) # Do some inference

        np.save('true.npy',true)
        np.save('predicted.npy',predicted)

