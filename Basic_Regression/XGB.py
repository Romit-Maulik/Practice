import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from scipy.stats import mode

import sys
np.set_printoptions(threshold=sys.maxsize)
np.random.seed(10)


class XGB_Class():
    def __init__(self, trainFilename, resultsDir):
        # assert len(trainFilenames) == len(testFilenames)
        self.resultsDir = resultsDir
        #ntrees = 1000
        self.trainFilename = trainFilename

        self.load_data()
        self.preprocess_data()
        self.model = MultiOutputRegressor(XGBRegressor())

        # Fit model
        self.model.fit(self.train_X_p,self.train_y_p)

    def load_data(self):
        filename = self.trainFilename
        if os.path.exists(filename):
            train_data = pd.read_csv(filename,encoding = "ISO-8859-1")

        out_df = train_data.iloc[:,-1].values.reshape(-1,1)
        inp_df = train_data.iloc[:,:-1]

        self.train_X = inp_df
        self.train_y = out_df

    def preprocess_data(self):
        self.preproc_X = Pipeline([('stdscaler', StandardScaler()),('minmax', MinMaxScaler(feature_range=(-1, 1)))])
        self.preproc_y = Pipeline([('stdscaler', StandardScaler()),('minmax', MinMaxScaler(feature_range=(-1, 1)))])
        self.train_X_p = self.preproc_X.fit_transform(self.train_X)#.as_matrix()
        self.train_y_p = self.preproc_y.fit_transform(self.train_y)#.as_matrix()

    def importances_rankings(self):
        importances = self.model.estimators_[0].feature_importances_
        # Tracking ranking
        indices = np.argsort(importances)[::-1]
        rankings = np.zeros(shape=(self.train_X_p.shape[1],),dtype='int')
        for f in range(self.train_X_p.shape[1]):
            rankings[int(indices[f])] = f+1

        return importances, rankings

def plot_ensemble_rankings(importance_tracker, ranking_tracker, variable_names, num_folds):

    importances = np.sum(importance_tracker,axis=0)/num_folds
    indices = np.argsort(importances)[::-1]
    # Plot individual feature importance
    plt.figure(figsize=(12,10))
    x = np.arange(len(variable_names))
    plt.barh(x,width=importances[indices])
    plt.yticks(x, [variable_names[indices[f]] for f in range(len(variable_names))])

    plt.ylabel('Feature',fontsize=24)
    plt.xlabel('Relative decisiveness',fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('importances/importances.png')
    plt.close()

    # Box plot
    modal_list = []
    for f in range(len(variable_names)):
        modal_list.append(mode(ranking_tracker[:,f]).mode[0])

    modal_indices = np.argsort(modal_list)[::-1]

    import seaborn as sns
    data_to_plot = []
    for f in range(importance_tracker.shape[1]):
        data_to_plot.append(ranking_tracker[:,modal_indices[f]])

    plt.figure(figsize=(18,10))
    box_plot = sns.boxplot(data=data_to_plot)
    box_plot.set_xticklabels([variable_names[modal_indices[f]] for f in range(len(variable_names))],fontsize=20)
    box_plot.set_ylabel('Rankings',fontsize=20)

    ax = box_plot.axes
    lines = ax.get_lines()
    categories = ax.get_xticks()

    for cat in categories:
        y = round(lines[4+cat*6].get_ydata()[0],1) 

        ax.text(
            cat, 
            y, 
            y, 
            ha='center', 
            va='center', 
            fontweight='bold', 
            size=20,
            color='white',
            bbox=dict(facecolor='#445A64'))

    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(rotation=90)
    plt.yticks(fontsize=20)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig('importances/relative_decisiveness.png')


if __name__ == '__main__':

    # Load variable names
    import argparse
    parser = argparse.ArgumentParser(description='Read csv file to run importance assessor')
    parser.add_argument('csv_file', metavar='csv_filename', type=str, help='csv data file path')
    parser.add_argument('num_folds', metavar='num_folds', type=str, help='number of folds')
    args = parser.parse_args()

    # Read data file 
    csv_df = pd.read_csv(args.csv_file,encoding = "ISO-8859-1")    
    # Record list of variables
    variable_names = csv_df.columns.tolist()[:-1]

    import os, glob
    if not os.path.exists('importances/'):
        os.mkdir('importances/')
    resultsDir = 'importances/'
    
    trainFilenames = []
    testFilenames = []
    pattern = 'folds/train_*.csv'
    trainFiles = glob.glob(pattern)

    # Plots for ranking/importance
    importance_tracker = []
    ranking_tracker = []

    for trainFilename in trainFiles:
        trainFilenames.append(trainFilename)
        xgb_temp = XGB_Class(trainFilename, resultsDir)
        importance, ranking = xgb_temp.importances_rankings()

        importance_tracker.append(importance)
        ranking_tracker.append(ranking)

    importance_tracker = np.asarray(importance_tracker)
    ranking_tracker = np.asarray(ranking_tracker)

    plot_ensemble_rankings(importance_tracker, ranking_tracker, variable_names, int(args.num_folds))
