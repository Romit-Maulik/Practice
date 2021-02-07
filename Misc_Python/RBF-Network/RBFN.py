"""
File: RBFN.py
Author: Octavio Arriaga
Email: arriaga.camargo@email.com
Github: https://github.com/oarriaga
Description: Minimal implementation of a radial basis function network
"""

import numpy as np


class RBFN(object):

    def __init__(self, hidden_shape, sigma=1.0):
        """ radial basis function network
        # Arguments
            input_shape: dimension of the input data
            e.g. scalar functions have should have input_dimension = 1
            hidden_shape: the number
            hidden_shape: number of hidden radial basis functions,
            also, number of centers.
        """
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point):
        return np.exp(-self.sigma*np.linalg.norm(center-data_point)**2)

    def activations(self,X):
        '''
        This function makes a list of activatinons for all the centers
        '''
        if len(np.shape(X)) == 1:
            dim = 1
        else:
        	dim = np.shape(X)[1]

        self.rho_h = np.zeros(shape=(self.hidden_shape),dtype='double')
        for center in range(self.hidden_shape):
            for data_point in range(len(X)):
                self.rho_h[center] = self.rho_h[center] + self._kernel_function(self.centers[center], X[data_point])
            self.rho_h[center] = self.rho_h[center]/(len(X)*((np.pi**(1/2)*self.sigma))**(dim))

    def confidence(self,xpred):

        self.confidence_vals = np.zeros(shape=(np.shape(xpred)[0]),dtype='double')

        max_act = 0.0
        for test_point in range(1,np.shape(xpred)[0]):
            act_sum = 0.0
            for center in range(1,self.hidden_shape):
                act_val = self._kernel_function(self.centers[center], xpred[test_point])
                max_act = max(max_act,act_val)
                self.confidence_vals[test_point] = self.confidence_vals[test_point] + act_val*self.rho_h[center]
                act_sum = act_sum + act_val
            self.confidence_vals[test_point] = self.confidence_vals[test_point]/(act_sum + 1.0 - max_act)

        return self.confidence_vals




    def _calculate_interpolation_matrix(self, X):
        """ Calculates interpolation matrix using a kernel_function
        # Arguments
            X: Training data
        # Input shape
            (num_data_samples, input_shape)
        # Returns
            G: Interpolation matrix
        """
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self._kernel_function(
                        center, data_point)
        return G

    def _select_centers(self, X):
        random_args = np.random.choice(len(X), self.hidden_shape)
        centers = X[random_args]
        return centers

    def fit(self, X, Y):
        """ Fits weights using linear regression
        # Arguments
            X: training samples
            Y: targets
        # Input shape
            X: (num_data_samples, input_shape)
            Y: (num_data_samples, input_shape)
        """
        self.centers = self._select_centers(X)
        G = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        """
        # Arguments
            X: test data
        # Input shape
            (num_test_samples, input_shape)
        """
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions
