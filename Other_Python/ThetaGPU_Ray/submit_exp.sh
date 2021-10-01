#!/bin/bash
#COBALT -A datascience
#COBALT -n 2
#COBALT -q full-node
#COBALT -t 60

START1="$(date +%s)"

source ray_initialization.sh

# source /lus/theta-fs0/software/thetagpu/conda/2021-06-28/mconda3/setup.sh
# conda activate /lus/eagle/projects/datascience/rmaulik/LSTM_Var_Prototype/AIAEDA

# # For running experiment
# python source/main.py
# python source/comparisons.py