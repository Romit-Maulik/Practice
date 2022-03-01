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
from sklearn.metrics import accuracy_score
import xgboost

from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)

from sklearn.pipeline import Pipeline
from sklearn import neighbors
from sklearn import tree
from sklearn import gaussian_process
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


import sklearn.dummy
import math
from sklearn.multioutput import MultiOutputClassifier

os.environ['KMP_DUPLICATE_LIB_OK']='True'
outputs = ['time']
fold = ['nfold', 'fold', 'id','percentage','nfold']

PP_OUT_FLAG = True
LOG_FLAG = False

class Classification():

    def __init__(self, trainFilename, testFilename, resultsDir, run_case=True):
        # assert len(trainFilenames) == len(testFilenames)
        self.resultsDir = resultsDir
        #ntrees = 1000
        self.trainFilename = trainFilename
        self.testFilename = testFilename
        self.classifiers = {
            'svm': MultiOutputClassifier(svm.SVC(kernel='rbf')),
            'gp': MultiOutputClassifier(gaussian_process.GaussianProcessClassifier()),
            'knn': MultiOutputClassifier(neighbors.KNeighborsClassifier(n_neighbors=5)),
            'dt': MultiOutputClassifier(tree.DecisionTreeClassifier()),
            'br': MultiOutputClassifier(ensemble.BaggingClassifier(n_jobs=-1)),
            'etr': MultiOutputClassifier(ensemble.ExtraTreesClassifier(n_jobs=-1)),
            'rfr': MultiOutputClassifier(ensemble.RandomForestClassifier(n_jobs=-1)),
            'abr': MultiOutputClassifier(ensemble.AdaBoostClassifier()),
            'gbr': MultiOutputClassifier(ensemble.GradientBoostingClassifier()),
            'xgb': MultiOutputClassifier(xgboost.XGBClassifier())
        }

        if trainFilename is not None and testFilename is not None:
            self.load_data()
            self.preprocess_data()
            for key in self.classifiers.keys():
                self.fit_model(key)
        else:
            print('Loading dummy classification class')


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
        self.train_y_p = self.train_y
        self.test_X_p = self.preproc_X.transform(self.test_X)#.as_matrix()
        self.test_y_p = self.test_y

    def build_model(self, model_type):
        start = time.time()
        model = self.classifiers[model_type]
        end = time.time()
        build_time = (end-start)
        return model, build_time

    def train_model(self, model, model_type):
        start = time.time()
        model.fit(self.train_X_p, self.train_y_p)
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
            acc = accuracy_score(y_true, y_pred) 
            result = [acc]
            results.append(result)
        res_df = pd.DataFrame(results)
        res_df.columns = ['acc']
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
            test_yhat = test_yhat_p
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

        Classification(trainFilename, testFilename, resultsDir)

