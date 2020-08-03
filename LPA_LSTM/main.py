import numpy as np
np.random.seed(10)

from lstm_archs import standard_lstm

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LSTM training of time-series data')
    parser.add_argument('--train', help='Do training', action='store_true') #Train a network or use a trained network for inference
    parser.add_argument('--lce',help='Use Lyapunov regularization',action='store_true')
    args = parser.parse_args()

    # Loading data
    data = np.load('SST_Train.npy').T
    # Initialize model
    lstm_model = standard_lstm(data,lce=args.lce)
    # Training model
    if args.train:
        lstm_model.train_model() # Train and exit
    else:
        data = np.load('SST_Test.npy').T
        true, predicted = lstm_model.model_inference(data) # Do some inference

