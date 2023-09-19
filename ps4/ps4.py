# Inital imports and setup

import os
import numpy as np

###################
# Helper function #
###################
def load_data(filepath):
    '''
    Load in the given csv filepath as a numpy array

    Parameters
    ----------
    filepath (string) : path to csv file

    Returns
    -------
        X, y (np.ndarray, np.ndarray) : (m, num_features), (m,) numpy matrices
    '''
    *X, y = np.genfromtxt(
        filepath,
        delimiter=',',
        skip_header=True,
        unpack=True,
    ) # default dtype: float
    X = np.array(X, dtype=float).T # cast features to int type
    return X, y.reshape((-1, 1))

data_filepath = 'housing_data.csv'
X, y = load_data(data_filepath)

def mean_squared_error(y_true, y_pred):
    '''
    Calculate mean squared error between y_pred and y_true.

    Parameters
    ----------
    y_true (np.ndarray) : (m, 1) numpy matrix consists of target value
    y_pred (np.ndarray) : (m, 1) numpy matrix consists of prediction
    
    Returns
    -------
        The mean squared error value.
    '''
    m = len(y_true)
    diffArr = y_true[:, 0] - y_pred[:, 0]
    diffArr = np.square(diffArr)
    return np.sum(diffArr) / m

def mean_absolute_error(y_true, y_pred):
    '''
    Calculate mean absolute error between y_pred and y_true.

    Parameters
    ----------
    y_true (np.ndarray) : (m, 1) numpy matrix consists of target value
    y_pred (np.ndarray) : (m, 1) numpy matrix consists of prediction
    
    Returns
    -------
        The mean absolute error value.
    '''
    m = len(y_true)
    diffArr = y_true[:, 0] - y_pred[:, 0]
    diffArr = np.abs(diffArr)
    return np.sum(diffArr) / m

def add_bias_column(X):
    '''
    Create a bias column and combined it with X.

    Parameters
    ----------
    X : (m, n) numpy matrix representing feature matrix
    
    Returns
    -------
        A (m, n + 1) numpy matrix with the first column consists of all 1
    '''
    length = len(X)
    bias = np.ones((length, 1))
    finalArr = np.append(bias, X, axis=1)
    return finalArr

def get_bias_and_weight(X, y, include_bias = True):
    '''
    Calculate bias and weights that give the best fitting line.

    Parameters
    ----------
    X (np.ndarray) : (m, n) numpy matrix representing feature matrix
    y (np.ndarray) : (m, 1) numpy matrix representing target values
    include_bias (boolean) : Specify whether the model should include a bias term
    
    Returns
    -------
        bias (float):
            If include_bias = True, return the bias constant. Else,
            return 0
        weights (np.ndarray):
            A (n, 1) numpy matrix representing the weight constant(s).
    '''
    # print(X)
    # print(y)
    # XtX = [14,21],[21,34]
    # (XtX)^-1 = [34/35, -3/5],[-3/5, 2/5]
    # (XtX)^-1 Xt = [-29/35,1/7,18/35],[3/5,0,-1/5]
    # (XtX)^-1 Xt y = [17/35, 6/5] = 0.485714
    if not include_bias:
        length = len(X)
        bias = np.zeros((length, 1))
        finalArr = np.append(bias, X, axis=1)
    else:
        X = add_bias_column(X)
    finalarr = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), y)
    if not include_bias:
        return (0, finalarr)
    ans = (finalarr[0][0], finalarr[1:])
    return ans

def get_prediction_linear_regression(X, y, include_bias = True):
    '''
    Calculate the best fitting line.

    Parameters
    ----------
    X (np.ndarray) : (m, n) numpy matrix representing feature matrix
    y (np.ndarray) : (m, 1) numpy matrix representing target values
    include_bias (boolean) : Specify whether to use bias term

    Returns
    -------
        A (m, 1) numpy matrix representing prediction values.
    '''
  
    # TODO: add your solution here and remove `raise NotImplementedError`
    raise NotImplementedError

  
if __name__ == "__main__":
    public_X, public_y = np.array([[1, 3], [2, 3], [3, 4]]), np.arange(4, 7).reshape((-1, 1))
    # ans = np.round(get_bias_and_weight(public_X, public_y)[0], 5)
    # print(ans)
    ans = np.round(get_bias_and_weight(public_X, public_y, False)[1], 2)
    print("output", ans)
    ans = np.round(np.array([[0.49], [1.20]]), 2)
    print("actual", ans)