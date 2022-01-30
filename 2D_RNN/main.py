import numpy as np
import pickle
np.random.seed(10)

from lstm_archs import grid_lstm

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LSTM training of time-series data')
    parser.add_argument('--train', help='Do training', action='store_true') #Train a network or use a trained network for inference
    args = parser.parse_args()

    # Loading data train and test
    z500_data = np.load('./Example_data/z500_cf.npy').T
    pwat_data = np.load('./Example_data/pwat_cf.npy').T
    train_data = [z500_data,pwat_data]

    z500_data_test = np.load('./Example_data/z500_cf_test.npy').T
    pwat_data_test = np.load('./Example_data/pwat_cf_test.npy').T
    test_data = [z500_data_test,pwat_data_test]

    # Forecast parameters
    resolution_list = [1,1] # the data element shape divided by resolution should be constant

    # These are input variable indices
    input_idx = [0,1]
    # Output variable indices
    output_idx = [0]
    # Horizons of input and forecasts
    input_horizon = 14 # These are days
    output_horizon = 7 # These are days
    
    # Initialize model
    lstm_model = grid_lstm(train_data,input_idx,output_idx,input_horizon,output_horizon,resolution_list)
    # Training model
    if args.train:
        lstm_model.train_model(batch_size=20,num_epochs=100) # Train and exit
    else:
        true, predicted = lstm_model.model_inference(test_data) # Do some inference

        with open('Predictions.pkl','wb') as f:
            pickle.dump([true,predicted],f)

        

