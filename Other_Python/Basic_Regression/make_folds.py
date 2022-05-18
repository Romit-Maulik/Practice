import sys

import numpy as np
np.random.seed(5)
np.set_printoptions(threshold=sys.maxsize)
from sklearn.model_selection import train_test_split

import pandas as pd
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Read csv file make folds')
    parser.add_argument('csv_file', metavar='csv_filename', type=str, help='csv data file path')
    parser.add_argument('num_folds', metavar='num_folds', type=str, help='number of folds')
    parser.add_argument('train_ratio', metavar='train_ratio', type=float, help='train-test ratio')
    args = parser.parse_args()

    # Read data file 
    csv_df = pd.read_csv(args.csv_file,encoding = "ISO-8859-1")
    csv_df = csv_df.apply(pd.to_numeric, errors='coerce') # Non-numeric values converted to NaN
    csv_df = csv_df.fillna(0.0)
    data = np.asarray(csv_df.values.tolist())

    # Record list of variables
    variables = csv_df.columns.tolist()[:-1]

    # Ready to do operations
    independent_vars = data[:,:-1].astype('double') # Last column is dependent
    dependent_vars = data[:,-1].astype('double')

    num_rows = np.shape(independent_vars)[0]
    num_vars = np.shape(independent_vars)[1]
    
    # Make folds path
    import os
    if not os.path.exists('folds/'):
        os.mkdir('folds/')

    # Split the data into training and testing sets
    for fold in range(int(args.num_folds)):
        independent_vars_train, independent_vars_test, dependent_vars_train, dependent_vars_test = train_test_split(independent_vars, dependent_vars.reshape(-1,1), train_size=float(args.train_ratio), random_state=fold)

        train_data = np.concatenate((independent_vars_train,dependent_vars_train),axis=-1)
        test_data = np.concatenate((independent_vars_test,dependent_vars_test),axis=-1)

        # Save the file
        np.savetxt('folds/train_'+f'{fold:02}'+'.csv',train_data,delimiter=',')
        np.savetxt('folds/test_'+f'{fold:02}'+'.csv',test_data,delimiter=',')
        
        