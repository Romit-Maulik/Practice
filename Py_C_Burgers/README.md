# Steps to reproduce
1. Run `cmake ..`  from within `build/` directory.
2. Run `make` from within `build/` directory.
3. `export PATH=/home/rmlans/anaconda3/envs/tf2_env/bin:$PATH`. Point to your specific python executable.
4. `export LD_LIBRARY_PATH=/home/rmlans/anaconda3/envs/tf2_env/lib:$LD_LIBRARY_PATH`. Add the python so file to the library path. This python should have numpy, tensorflow and matplotlib installed.

# Versions:
1. cmake 3.10.2
2. python 3.6.8 gcc 7.3.0
3. numpy 1.18.1
4. tensorflow 2.2.0
5. matplotlib 3.1.0

# What you should see

## Field evolution
![Fields](Field_evolution.png "Fields")

## Modal decomposition
![Modes](SVD_Eigenvectors.png "Modes")

## Forecasting the modal evolution in time (still rather poor but you get the idea)
![Forecasting Mode 0](Mode_0_prediction.png "Mode 0 prediction")

![Forecasting Mode 1](Mode_1_prediction.png "Mode 1 prediction")

![Forecasting Mode 2](Mode_2_prediction.png "Mode 2 prediction")
