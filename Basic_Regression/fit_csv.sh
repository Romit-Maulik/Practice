# 
if [ "$1" == "-h" ]; then
	echo ""
    echo "A bash script to fit a csv file and generate metrics/feature decisiveness"
    echo "Note that csv file is assumed to have only one dependent variable (as last column)"
    echo "csv file should also have header with variable names"
    echo ""
    echo "Requires python 3.6.8, tensorflow 1.14, numpy, scikit-learn, xgboost, pandas, matplotlib"
    echo ""
    echo "Usage:"
    echo "bash fit_csv.sh csvfilename.csv num_folds"
    echo ""
    echo "Returns:"
    echo "1. 'folds/' which has all the training and test fold csv files"
    echo "2. 'results/' which outputs the raw predictions of the fit models on the test data"
    echo "3. 'metrics/' which has outputs of the different metrics in csv and image format"
    echo "4. 'importances/' which has images of the ensemble rankings for each fold and the relative decisiveness"
    echo ""
    echo "Author: Romit Maulik"
    exit 0
else
    if [ $# -eq 2 ]; then
    	
    	rm -rf folds/
    	rm -rf importances/
    	rm -rf results/
    	rm -rf metrics/

        python make_folds.py $1 $2
        python run_regressors.py
        python parse_results.py $2
        python XGB.py $1 $2
        
        exit 0
    else
        echo "Wrong number of arguments, abort; see help with -h"
        exit 1
    fi
fi

