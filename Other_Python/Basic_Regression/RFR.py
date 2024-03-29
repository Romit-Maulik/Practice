import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from scipy.stats import mode
import shap
from sklearn import ensemble
from sklearn.metrics import r2_score

import sys
np.set_printoptions(threshold=sys.maxsize)
np.random.seed(10)


class RFR_Class():
    def __init__(self, trainFilename, resultsDir, variable_names, fold_num):
        # assert len(trainFilenames) == len(testFilenames)
        self.resultsDir = resultsDir
        #ntrees = 1000
        self.trainFilename = trainFilename
        self.variable_names = variable_names
        self.fold_num = fold_num

        self.load_data()
        self.preprocess_data()
        self.model = ensemble.RandomForestRegressor(n_estimators=100) # Change this number for different number trees

        # Fit model
        self.model.fit(self.train_X_p,self.train_y_p)

        # Get directions of variable importance
        self.variable_directions()

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

    def permutation_importances(self):
        baseline = r2_score(self.model.predict(self.train_X_p),self.train_y_p)
        imp = []
        temp_df = pd.DataFrame(data=self.train_X_p,columns=self.variable_names)
        for col in temp_df.columns:
            save = temp_df[col].copy()
            temp_df[col] = np.random.permutation(temp_df[col])
            m = r2_score(self.model.predict(temp_df), self.train_y_p)
            temp_df[col] = save
            imp.append(baseline - m)
        return np.array(imp)

    def importances_rankings(self):

        importances = self.permutation_importances()#.model.feature_importances_
        # Tracking ranking
        indices = np.argsort(importances)[::-1]
        rankings = np.zeros(shape=(self.train_X_p.shape[1],),dtype='int')
        for f in range(self.train_X_p.shape[1]):
            rankings[int(indices[f])] = f+1

        return importances, rankings

    def variable_directions(self):
        # explain the model's predictions using SHAP
        # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
        explainer = shap.TreeExplainer(self.model)

        # Make pandas dataframe for visualization
        # https://github.com/slundberg/shap/issues/153
        temp_dataframe = pd.DataFrame(data=self.train_X_p,columns=self.variable_names)
        shap_values = explainer.shap_values(self.train_X_p)
        shap.save_html('importances/force_plots/Fold_'+str(self.fold_num)+'_force_plot.html',shap.force_plot(explainer.expected_value, shap_values, temp_dataframe))

        plt.figure()
        shap.summary_plot(shap_values, temp_dataframe,show=False)
        plt.tight_layout()
        plt.savefig('importances/summary_plots/Fold_'+str(self.fold_num)+'_summary_plot.png',bbox_inches='tight')
        plt.close()


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
    plt.close()


if __name__ == '__main__':

    # Load variable names
    import argparse
    parser = argparse.ArgumentParser(description='Read csv file to run importance assessor')
    parser.add_argument('csv_file', metavar='csv_filename', type=str, help='csv data file path')
    parser.add_argument('num_folds', metavar='num_folds', type=str, help='number of folds')
    parser.add_argument('num_trees', metavar='num_trees', type=str, help='number of decision trees')
    args = parser.parse_args()

    # Read data file 
    csv_df = pd.read_csv(args.csv_file,encoding = "ISO-8859-1")
    csv_df = csv_df.apply(pd.to_numeric, errors='coerce') # Non-numeric values converted to NaN
    csv_df = csv_df.fillna(0.0)   
    # Record list of variables
    variable_names = csv_df.columns.tolist()[:-1]

    import os
    import glob
    if not os.path.exists('importances/'):
        os.mkdir('importances/')
        os.mkdir('importances/force_plots/')
        os.mkdir('importances/summary_plots/')
    resultsDir = 'importances/'

    trainFilenames = []
    testFilenames = []
    pattern = 'folds/train_*.csv'
    trainFiles = glob.glob(pattern)

    # Plots for ranking/importance
    importance_tracker = []
    ranking_tracker = []

    fold_num = 0
    for trainFilename in trainFiles:
        trainFilenames.append(trainFilename)
        gbr_temp = RFR_Class(trainFilename, resultsDir, variable_names, fold_num)
        importance, ranking = gbr_temp.importances_rankings()

        importance_tracker.append(importance)
        ranking_tracker.append(ranking)
        fold_num = fold_num + 1

    importance_tracker = np.asarray(importance_tracker)
    ranking_tracker = np.asarray(ranking_tracker)

    plot_ensemble_rankings(importance_tracker, ranking_tracker, variable_names, int(args.num_folds))

    # Make SHAP rankings for consistency
    # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
    model = ensemble.RandomForestRegressor(n_estimators = int(args.num_trees))
    # Load data
    out_df = csv_df.iloc[:,-1].values.reshape(-1,1)
    inp_df = csv_df.iloc[:,:-1] 

    preproc_X = Pipeline([('stdscaler', StandardScaler()),('minmax', MinMaxScaler(feature_range=(-1, 1)))])
    preproc_y = Pipeline([('stdscaler', StandardScaler()),('minmax', MinMaxScaler(feature_range=(-1, 1)))])
    inp_df = preproc_X.fit_transform(inp_df)#.as_matrix()
    out_df = preproc_y.fit_transform(out_df)#.as_matrix()

    # Fit model
    model.fit(inp_df,out_df)
    explainer = shap.TreeExplainer(model)

    # Make pandas dataframe for visualization
    # https://github.com/slundberg/shap/issues/153
    temp_dataframe = pd.DataFrame(data=inp_df,columns=variable_names)
    shap_values = explainer.shap_values(inp_df)

    plt.figure()
    shap.summary_plot(shap_values,temp_dataframe,plot_type='bar',show=False)
    plt.tight_layout()
    plt.savefig('importances/SHAP_importances.png',bbox_inches='tight')
    plt.close()