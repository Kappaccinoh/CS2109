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
    if not include_bias:
        length = len(X)
        bias = np.zeros((length, 1))
        finalArr = np.append(bias, X, axis=1)
    else:
        X = add_bias_column(X)
    finalarr = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), y)
    if not include_bias:
        # ans = (0, [[w1],[w2],...])
        return (0, finalarr)
    ans = (finalarr[0][0], finalarr[1:])
    # ans = (bias = w0, [[w1], [w2],...])
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
    length = len(X)
    w = get_bias_and_weight(X,y,include_bias)
    bias = w[0]
    biasCol = np.full((length, 1), bias)

    # X = (num entries, num features)
    # w = (num features)
    # res = Xw + bias
    res = np.matmul(X, w[1:])+ biasCol
    return res

def gradient_descent_one_variable(x, y, lr = 1e-5, number_of_epochs = 250):
    '''
    Approximate bias and weight that gave the best fitting line.

    Parameters
    ----------
    x (np.ndarray) : (m, 1) numpy matrix representing a feature column
    y (np.ndarray) : (m, 1) numpy matrix representing target values
    lr (float) : Learning rate
    number_of_epochs (int) : Number of gradient descent epochs
    
    Returns
    -------
        bias (float):
            The bias constant
        weight (float):
            The weight constant
        loss (list):
            A list where the i-th element denotes the MSE score at i-th epoch.
    '''
    # Do not change
    bias = 0
    weight = 0
    loss = []

    length = len(x)

    for i in range(number_of_epochs):
        biasMatrix = np.full((length, 1), bias)
        weightMatrix = np.full((length, 1), weight)

        partial = np.sum(biasMatrix + weightMatrix * x - y) / length
        newBias = bias - lr * partial

        partial = np.sum((biasMatrix + weightMatrix * x - y) * x) / length
        newWeight = weight - lr * partial

        bias = newBias
        weight = newWeight

        '''
        y_true (np.ndarray) : (m, 1) numpy matrix consists of target value
        y_pred (np.ndarray) : (m, 1) numpy matrix consists of prediction
        '''
        y_predicted = np.full((length, 1), 0)
        biasMatrix = np.full((length, 1), bias)
        weightMatrix = np.full((length, 1), weight)
        y_predicted = y_predicted + biasMatrix + weightMatrix * x
        y_true = y
        mseScore = mean_squared_error(y_true, y_predicted)
        loss.append(mseScore)
    
    return bias, weight, loss

def gradient_descent_multi_variable(X, y, lr = 1e-5, number_of_epochs = 250):
    '''
    Approximate bias and weight that gave the best fitting line.

    Parameters
    ----------
    X (np.ndarray) : (m, n) numpy matrix representing feature matrix
    y (np.ndarray) : (m, 1) numpy matrix representing target values
    lr (float) : Learning rate
    number_of_epochs (int) : Number of gradient descent epochs
    
    Returns
    -------
        bias (float):
            The bias constant
        weights (np.ndarray):
            A (n, 1) numpy matrix that specifies the weight constants.
        loss (list):
            A list where the i-th element denotes the MSE score at i-th epoch.
    '''
    length = len(X)
    n = len(X[0])

    # Do not change
    bias = 0
    weights = np.full((n, 1), 0).astype(float)
    loss = []

    for i in range(number_of_epochs):
        # updating bias
        biasMatrix = np.full((length, 1), bias)
        partial = np.sum(biasMatrix + np.sum(np.matmul(X, weights), axis=1).reshape(length, 1) - y) / length
        newBias = bias - lr * partial

        partial0 = biasMatrix + np.sum(np.matmul(X, weights), axis=1).reshape(length, 1) - y
        partial1 = np.matmul(np.array(X).transpose(), partial0) / length
        newWeight = weights - lr * partial1

        bias = newBias
        weights = newWeight
        '''
        y_true (np.ndarray) : (m, 1) numpy matrix consists of target value
        y_pred (np.ndarray) : (m, 1) numpy matrix consists of prediction
        '''
        y_predicted = np.full((length, 1), 0)
        biasMatrix = np.full((length, 1), bias)
        y_predicted = y_predicted + biasMatrix + np.matmul(X, weights)
        y_true = np.array(y)
        mseScore = mean_squared_error(y_true, y_predicted)
        loss.append(mseScore)
    
    return bias, weights, loss

def create_polynomial_matrix(X, power = 2):
    '''
    Create a polynomial matrix.
    
    Parameters
    ----------
    X: (m, 1) numpy matrix

    Returns
    -------
        A (m, power) numpy matrix where the i-th element denotes
            X raised to the power of i.
    '''
    length = len(X)
    a = np.tile(X, power)
    a = np.cumprod(a, axis=1)
    return a

def get_prediction_poly_regression(X, y, power = 2, include_bias = True):
    '''
    Calculate the best polynomial line.

    Parameters
    ----------
    X (np.ndarray) : (m, 1) numpy matrix representing feature matrix
    y (np.ndarray) : (m, 1) numpy matrix representing target values
    power (int) : Specify the degree of the polynomial
    include_bias (boolean) : Specify whether to use bias term

    Returns
    -------
        A (m, 1) numpy matrix representing prediction values.
    '''
    polyMatrix = create_polynomial_matrix(X, power)
    linreg = get_prediction_linear_regression(polyMatrix, y, include_bias)
    return linreg

def feature_scaling(X):
    '''
    Mean normalized each feature column.

    Parameters
    ----------
    X (np.ndarray) : (m, n) numpy matrix representing feature matrix

    Returns
    -------
        A (m, n) numpy matrix where each column has been mean-normalized.
    '''
    m = len(X)
    n = len(X[0])

    meanMatrix = np.mean(X, axis=0)
    meanMatrix = meanMatrix.reshape((1,-1))
    meanMatrix = np.repeat(meanMatrix, m, axis=0)

    stdMatrix = np.std(X, axis=0)
    stdMatrix = stdMatrix.reshape((1,-1))
    stdMatrix = np.repeat(stdMatrix, m, axis=0)

    a = (X - meanMatrix) / stdMatrix
    return a

def find_number_of_epochs(X, y, lr, delta_loss):
    '''
    Do gradient descent until convergence and return number of epochs
    required.

    Parameters
    ----------
    X (np.ndarray) : (m, n) numpy matrix representing feature matrix
    y (np.ndarray) : (m, 1) numpy matrix representing target values
    lr (float) : Learning rate
    delta_loss (float) : Termination criterion
    
    Returns
    -------
        bias (float):
            The bias constant
        weights (np.ndarray):
            A (n, 1) numpy matrix that specifies the weight constants.
        num_of_epochs (int):
            Number of epochs to reach convergence.
        current_loss (float):
            The loss value obtained after convergence.
    '''
    # Do not change
    bias = 0
    weights = np.full((X.shape[1], 1), 0).astype(float)
    num_of_epochs = 0
    previous_loss = 1e14
    current_loss = -1e14
    
    length = len(X)
    n = len(X[0])

    while abs(previous_loss - current_loss) >= delta_loss:
        # updating bias
        biasMatrix = np.full((length, 1), bias)
        partial = np.sum(biasMatrix + np.sum(np.matmul(X, weights), axis=1).reshape(length, 1) - y) / length
        newBias = bias - lr * partial

        partial0 = biasMatrix + np.sum(np.matmul(X, weights), axis=1).reshape(length, 1) - y
        partial1 = np.matmul(np.array(X).transpose(), partial0) / length
        newWeight = weights - lr * partial1

        bias = newBias
        weights = newWeight
        '''
        y_true (np.ndarray) : (m, 1) numpy matrix consists of target value
        y_pred (np.ndarray) : (m, 1) numpy matrix consists of prediction
        '''
        y_predicted = np.full((length, 1), 0)
        biasMatrix = np.full((length, 1), bias)
        y_predicted = y_predicted + biasMatrix + np.matmul(X, weights)
        y_true = np.array(y)
        mseScore = mean_squared_error(y_true, y_predicted)

        previous_loss = current_loss
        current_loss = mseScore
        num_of_epochs += 1
    
    return bias, weights, num_of_epochs, current_loss


if __name__ == "__main__":
    X = [
        [4,5,2,3,4],
        [2,3,4,5,6],
        [4,5,6,7,8],
        [3,7,2,2,6]
    ]

    feature_scaling(X)