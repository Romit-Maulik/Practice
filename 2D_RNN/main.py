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
    blh_data = np.load('./POD/train_blh_coeffs.npy').T[:,:3]
    q250_data = np.load('./POD/train_q250_coeffs.npy').T[:,:3]
    q850_data = np.load('./POD/train_q850_coeffs.npy').T[:,:3]
    t250_data = np.load('./POD/train_t250_coeffs.npy').T[:,:3]
    t850_data = np.load('./POD/train_t850_coeffs.npy').T[:,:3]
    tcwv_data = np.load('./POD/train_tcwv_coeffs.npy').T[:,:3]
    u250_data = np.load('./POD/train_u250_coeffs.npy').T[:,:3]
    u850_data = np.load('./POD/train_u850_coeffs.npy').T[:,:3]
    v250_data = np.load('./POD/train_v250_coeffs.npy').T[:,:3]
    v850_data = np.load('./POD/train_v850_coeffs.npy').T[:,:3]
    
    train_data = [blh_data,q250_data,q850_data,t250_data,t850_data,
                    tcwv_data,u250_data,u850_data,v250_data,v850_data]


    blh_data_test = np.load('./POD/test_blh_coeffs.npy').T[:,:3]
    q250_data_test = np.load('./POD/test_q250_coeffs.npy').T[:,:3]
    q850_data_test = np.load('./POD/test_q850_coeffs.npy').T[:,:3]
    t250_data_test = np.load('./POD/test_t250_coeffs.npy').T[:,:3]
    t850_data_test = np.load('./POD/test_t850_coeffs.npy').T[:,:3]
    tcwv_data_test = np.load('./POD/test_tcwv_coeffs.npy').T[:,:3]
    u250_data_test = np.load('./POD/test_u250_coeffs.npy').T[:,:3]
    u850_data_test = np.load('./POD/test_u850_coeffs.npy').T[:,:3]
    v250_data_test = np.load('./POD/test_v250_coeffs.npy').T[:,:3]
    v850_data_test = np.load('./POD/test_v850_coeffs.npy').T[:,:3]

    test_data = [blh_data_test,q250_data_test,q850_data_test,t250_data_test,t850_data_test,
                    tcwv_data_test,u250_data_test,u850_data_test,v250_data_test,v850_data_test]

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(blh_data[:,0])
    # plt.plot(blh_data_test[:,0])
    # plt.show()
    # exit()

    # Forecast parameters
    resolution_list = [1,1,1,1,1,1,1,1,1,1] # the data element shape divided by resolution should be constant

    # These are input variable indices
    input_idx = [0,1,2,3,4,5,6,7,8,9]
    # Output variable indices
    output_idx = [6,7,8,9]
    # Horizons of input and forecasts
    input_horizon = 14 # These are days
    output_horizon = 3 # These are days

    # Training model
    if args.train:
        # Initialize model
        lstm_model = grid_lstm(train_data,input_idx,output_idx,input_horizon,output_horizon,resolution_list)
        lstm_model.train_model(batch_size=64,num_epochs=2000) # Train and exit
    else:
        lstm_model = grid_lstm(train_data.copy(),input_idx,output_idx,input_horizon,output_horizon,resolution_list)
        true, predicted = lstm_model.model_inference(train_data) # Do some inference

        with open('Train_Predictions.pkl','wb') as f:
            pickle.dump([true,predicted],f)
        f.close()

        true, predicted = lstm_model.model_inference(test_data) # Do some inference
        with open('Test_Predictions.pkl','wb') as f:
            pickle.dump([true,predicted],f)
        f.close()

        

