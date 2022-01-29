#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import sys

import numpy as np
np.random.seed(5)
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import glob
import os
import time
import json
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn import neural_network
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation
#from keras.layers.advanced_activations import PReLU, SReLU, LeakyReLU
import xgboost

from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)

from sklearn.pipeline import Pipeline
from sklearn import neighbors
from sklearn import tree
from sklearn import gaussian_process
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


#from keras.layers import Input, Dense
#from keras.models import Model
#import keras.backend as K
from sklearn.metrics import r2_score, mean_squared_error
import sklearn.dummy
import math
from sklearn.multioutput import MultiOutputRegressor

import tensorflow as tf
tf.random.set_seed(10)
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=50)

# Methods
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def coeff_determination(y_pred, y_true): #Order of function inputs is important here        
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

os.environ['KMP_DUPLICATE_LIB_OK']='True'
outputs = ['time']
fold = ['nfold', 'fold', 'id','percentage','nfold']

PP_OUT_FLAG = True
LOG_FLAG = False

class Regression():

    def __init__(self, trainFilename, testFilename, resultsDir, run_case=True):
        # assert len(trainFilenames) == len(testFilenames)
        self.resultsDir = resultsDir
        #ntrees = 1000
        self.trainFilename = trainFilename
        self.testFilename = testFilename
        self.regressors = {
            'lm': MultiOutputRegressor(linear_model.LinearRegression()),
            'rg': MultiOutputRegressor(linear_model.Ridge()),
            'svm': MultiOutputRegressor(svm.SVR(kernel='rbf')),
            'gp': MultiOutputRegressor(gaussian_process.GaussianProcessRegressor()),
            'knn': MultiOutputRegressor(neighbors.KNeighborsRegressor(n_neighbors=5)),
            'dt': MultiOutputRegressor(tree.DecisionTreeRegressor()),
            'br': MultiOutputRegressor(ensemble.BaggingRegressor(n_jobs=-1)),
            'etr': MultiOutputRegressor(ensemble.ExtraTreesRegressor(n_jobs=-1)),
            'rfr': MultiOutputRegressor(ensemble.RandomForestRegressor(n_jobs=-1)),
            'abr': MultiOutputRegressor(ensemble.AdaBoostRegressor()),
            'gbr': MultiOutputRegressor(ensemble.GradientBoostingRegressor()),
            'xgb': MultiOutputRegressor(xgboost.XGBRegressor()),
            'dl': None
        }

        if trainFilename is not None and testFilename is not None:
            self.load_data()
            self.preprocess_data()
            for key in self.regressors.keys():
                self.fit_model(key)
        else:
            print('Loading dummy regression class')


    def load_data(self):
        filename = self.trainFilename
        print(self.trainFilename)
        if os.path.exists(filename):
            train_data = pd.read_csv(filename,header=None,encoding = "ISO-8859-1")
        filename = self.testFilename
        if os.path.exists(filename):
            test_data1 = pd.read_csv(filename,header=None,encoding = "ISO-8859-1")

        out_df = train_data.iloc[:,-1].values.reshape(-1,1)
        inp_df = train_data.iloc[:,:-1]

        test_out_df1 = test_data1.iloc[:,-1].values.reshape(-1,1)
        test_inp_df1 = test_data1.iloc[:,:-1]

        self.train_X = inp_df
        self.train_y = out_df
        self.test_X = test_inp_df1
        self.test_y = test_out_df1

    def preprocess_data(self):
        self.preproc_X = Pipeline([('stdscaler', StandardScaler()),('minmax', MinMaxScaler(feature_range=(-1, 1)))])
        self.preproc_y = Pipeline([('stdscaler', StandardScaler()),('minmax', MinMaxScaler(feature_range=(-1, 1)))])
        self.train_X_p = self.preproc_X.fit_transform(self.train_X)#.as_matrix()
        self.train_y_p = self.preproc_y.fit_transform(self.train_y)#.as_matrix()
        self.test_X_p = self.preproc_X.transform(self.test_X)#.as_matrix()
        self.test_y_p = self.preproc_y.transform(self.test_y)#.as_matrix()

    def build_model(self, model_type):
        start = time.time()
        if model_type != 'dl':
            model = self.regressors[model_type]
        else:
            tf.keras.backend.clear_session()
            # tf.reset_default_graph()
            nunits = 200
            # design network
            model = Sequential()
            model.add(Dense(nunits, activation='tanh', input_shape=(self.train_X.shape[1],)))
            model.add(Dense(nunits, activation='tanh'))
            model.add(Dense(nunits, activation='tanh'))
            model.add(Dense(self.train_y.shape[1], activation='linear'))
            model.compile(loss='mse', optimizer='adam',metrics=[coeff_determination])
            model.summary()
        end = time.time()
        build_time = (end-start)
        return model, build_time

    def train_model(self, model, model_type):
        start = time.time()
        if model_type != 'dl':
            model.fit(self.train_X_p, self.train_y_p)
        else:
            model.fit(self.train_X_p, self.train_y_p, epochs=1000, batch_size=16, validation_split=0.1, verbose=1, callbacks=[early_stopping_monitor], shuffle=True)
        end = time.time()
        training_time = (end - start)
        return model, training_time

    def test_model(self, model):
        start = time.time()
        test_yhat_p = model.predict(self.test_X_p)
        end = time.time()
        inference_time = (end - start)
        return test_yhat_p , inference_time

    def compute_metric(self, test_y, test_yhat):
        results = []
        test_y = test_y.reshape(-1,1)
        test_yhat = test_yhat
        for out_index in range(test_y.shape[1]):
            y_true = test_y[:,out_index]
            y_pred = test_yhat[:,out_index]
            r2 = r2_score(y_true, y_pred) 
            evs = explained_variance_score(y_true, y_pred) 
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            rho = np.corrcoef(y_true, y_pred)[0][1]
            # mape = mean_absolute_percentage_error(y_true, y_pred) 
            result = [r2, rho, evs, mae, rmse]
            results.append(result)
        res_df = pd.DataFrame(results)
        res_df.columns = ['r2','rho','evs', 'mae', 'rmse']
        return(res_df)

    def fit_model(self, model_type):

        res_dict = {}
        outputFilename = os.path.basename(self.trainFilename).replace('train', 'meta_%s' % (model_type))
        output_base = os.path.splitext(outputFilename)[0]
        outputFilename = '%s/%s.json' % (self.resultsDir,output_base)

        if not os.path.exists(outputFilename):
            model, build_time = self.build_model(model_type)
            model, train_time = self.train_model(model, model_type)
                       
            test_yhat_p , inference_time = self.test_model(model)
            test_yhat = self.preproc_y.inverse_transform(test_yhat_p)
            res_df = self.compute_metric(self.test_y,test_yhat)
            
            res_dict['build_time'] = build_time
            res_dict['train_time'] = train_time
            res_dict['inference_time'] = inference_time
            res_dict['model'] = model_type

            outputFilename = os.path.basename(self.trainFilename).replace('train', 'meta_%s' % (model_type))
            output_base = os.path.splitext(outputFilename)[0]
            outputFilename = '%s/%s.json' % (self.resultsDir,output_base)
            
            with open(outputFilename, 'w') as fp:
                json.dump(res_dict, fp)

            output_base = output_base.replace('meta', 'pred')
            outputFilename = '%s/%s.csv' % (self.resultsDir,output_base)
            np.savetxt(outputFilename, test_yhat, delimiter=",")

            output_base = output_base.replace('pred', 'metric')
            outputFilename = '%s/%s.csv' % (self.resultsDir,output_base)
            res_df.to_csv(outputFilename)

if __name__ == '__main__':
    import os
    if not os.path.exists('results/'):
        os.mkdir('results/')
    resultsDir = 'results/'
    
    trainFilenames = []
    testFilenames = []
    pattern = 'folds/train_*.csv'
    trainFiles = glob.glob(pattern)
    
    for trainFilename in trainFiles:
        trainFilenames.append(trainFilename)
        testFilename = trainFilename.replace('train', 'test')
        testFilenames.append(testFilename)

        Regression(trainFilename, testFilename, resultsDir)

