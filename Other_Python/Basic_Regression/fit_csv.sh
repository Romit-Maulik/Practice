# 
if [ "$1" == "-h" ]; then
    echo ""
    echo "A bash script to fit multiple csv files and generate metrics/feature decisiveness"
    echo "Note that csv files are assumed to have only one dependent variable (as last column)"
    echo "csv file should also have header with variable names"
    echo "All csv files should be in one folder"
    echo ""
    echo "Requires python 3.6.8, tensorflow 1.14, numpy, scikit-learn, xgboost, pandas, matplotlib"
    echo ""
    echo "Usage:"
    echo "bash fit_csv.sh csvfolder num_folds train_ratio num_trees" # Order of arguments
    echo ""
    echo "Returns one folder for each csv file with the following subfolders:"
    echo "1. 'folds/' which has all the training and test fold csv files"
    echo "2. 'results/' which outputs the raw predictions of the fit models on the test data"
    echo "3. 'metrics/' which has outputs of the different metrics in csv and image format"
    echo "4. 'importances/' which has images of the ensemble rankings for each fold and the relative decisiveness"
    echo ""
    echo "Author: Romit Maulik"
    exit 0
else
    if [ $# -eq 4 ]; then
    	
        # Find the csv files and copy to individual directories
        cd $1

        for f in *.csv
        do
            cp $f ../            
        done

        cd ../
        
        for f in *.csv
        do
            folder_name=$(basename ${f} .csv)

            if [ -d "${folder_name}" ]
            then
                rm -rf "${folder_name}/folds/"
                rm -rf "${folder_name}/importances/"
                rm -rf "${folder_name}/results/"
                rm -rf "${folder_name}/metrics/"
            else
                mkdir "${folder_name}"
            fi

            python make_folds.py $f $2 $3 # $2 is the number of folds and $f is the name of the file
            python run_regressors.py $4
            python parse_results.py $2 $4
            python RFR.py $f $2 $4

            mv $f "${folder_name}/"
            mv folds "${folder_name}/"
            mv results "${folder_name}/"
            mv metrics "${folder_name}/"
            mv importances "${folder_name}/"

        done

        exit 0
    else
        echo "Wrong number of arguments, abort; see help with -h"
        exit 1
    fi
fi

