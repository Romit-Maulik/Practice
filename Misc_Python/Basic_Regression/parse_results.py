import numpy as np
np.random.seed(5)
import matplotlib.pyplot as plt
import argparse
from run_regressors import Regression

parser = argparse.ArgumentParser(description='Parse results for all folds')
parser.add_argument('num_folds', metavar='num_folds', type=str, help='csv data file path')
args = parser.parse_args()

# Make metrics path
import os
if not os.path.exists('metrics/'):
    os.mkdir('metrics/')

reg_class = Regression(None,None,None,run_case=False)
key_list = list(reg_class.regressors.keys())

metric_list = ['r2','rho','evs','mae','rmse']
num_methods = len(key_list)
num_metrics = len(metric_list)

metric_matrix = np.zeros(shape=(int(args.num_folds),num_methods,num_metrics))

for fold in range(int(args.num_folds)):
    method = 0
    for key in key_list:
        fname = 'metric_'+key+'_'+f'{fold:02}'
        temp_load = np.loadtxt('results/'+fname+'.csv',delimiter=',',usecols=[1,2,3,4,5],skiprows=1)

        metric_matrix[fold,method,0] = temp_load[0]
        metric_matrix[fold,method,1] = temp_load[1]
        metric_matrix[fold,method,2] = temp_load[2]
        metric_matrix[fold,method,3] = temp_load[3]
        metric_matrix[fold,method,4] = temp_load[4]

        method = method + 1

# Box plots
show_box_plots = True
if show_box_plots:
    import seaborn as sns
    for metric in range(num_metrics):

        data_to_plot = []
        for method in range(0,num_methods):
            data_to_plot.append(metric_matrix[:,method,metric])

        box_plot = sns.boxplot(data=data_to_plot)
        box_plot.set_xticklabels(key_list,fontsize=20)
        box_plot.set_ylabel(metric_list[metric],fontsize=20)

        ax = box_plot.axes
        lines = ax.get_lines()
        categories = ax.get_xticks()

        for cat in categories:
            y = round(lines[4+cat*6].get_ydata()[0],1) 

        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xticks(rotation=90)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig('metrics/'+metric_list[metric]+'.png')
        plt.clf()

# Write out average and std of metric_matrix
metric_means = np.zeros(shape=(num_metrics,num_methods))
metric_stds = np.zeros(shape=(num_metrics,num_methods))

for metric in range(num_metrics):
    for method in range(num_methods):
        metric_means[metric,method] = np.mean(metric_matrix[:,method,metric],axis=0)
        metric_stds[metric,method] = np.std(metric_matrix[:,method,metric],axis=0)

# Save metric matrices
import pandas as pd
mean_df = pd.DataFrame(metric_means,columns=key_list,index=metric_list)
std_df = pd.DataFrame(metric_stds,columns=key_list,index=metric_list)
mean_df.to_csv('metrics/Metric_Means.csv')
std_df.to_csv('metrics/Metric_stds.csv')
