# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:36:48 2020

@author: matth
"""

import autograd.numpy as np
#%% Kernel operations

# Returns the norm of the pairwise difference
def norm_matrix(matrix_1, matrix_2):
    norm_square_1 = np.sum(np.square(matrix_1), axis = 1)
    norm_square_1 = np.reshape(norm_square_1, (-1,1))
    
    norm_square_2 = np.sum(np.square(matrix_2), axis = 1)
    norm_square_2 = np.reshape(norm_square_2, (-1,1))
    
    d1=matrix_1.shape
    d2=matrix_2.shape

    if d1[1]!=d2[1]:
        matrix_1=np.transpose(matrix_1)
    
    inner_matrix = np.matmul(matrix_1, np.transpose(matrix_2))
    
    norm_diff = -2 * inner_matrix + norm_square_1 + np.transpose(norm_square_2)
    
    return norm_diff

# Returns the pairwise inner product
def inner_matrix(matrix_1, matrix_2):
    d1=matrix_1.shape
    d2=matrix_2.shape

    if d1[1]!=d2[1]:
        matrix_1=np.transpose(matrix_1)
    return np.matmul(matrix_1, np.transpose(matrix_2))


if __name__ == '__main__':
    print('This is the matrix operations file')