print("From python:")
print("Loading module")

import os,sys
HERE = os.getcwd()
sys.path.insert(0,HERE)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("From python:")
print('My tensorflow version is:',tf.__version__)

def python_function(i):

    print('From python: ')
    print('*****************  Within python method now  *********************')
    
    print('From python: ')
    print('The type of argument is:',i.dtype)

    return 2*i
