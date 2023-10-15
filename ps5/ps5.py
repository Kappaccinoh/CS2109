# Initial imports and setup

import numpy as np
import os
import pandas as pd

from sklearn import svm
from sklearn import model_selection

# Read credit card data into a Pandas dataframe for large tests

dirname = os.getcwd()
credit_card_data_filepath = os.path.join(dirname, 'credit_card.csv')
restaurant_data_filepath = os.path.join(dirname, 'restaurant_data.csv')

credit_df = pd.read_csv(credit_card_data_filepath)
X = credit_df.values[:, :-1]
y = credit_df.values[:, -1:]

# 1.2.1 Random undersampling in practice
def random_undersampling(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Given credit card dataset with 0 as the majority for 'Class' column. Returns credit card data with two classes
    having the same shape, given raw data read from csv.

    Parameters
    ----------
    df: pd.DataFrame
        The potentially imbalanced dataset.

    Returns
    -------
    Undersampled dataset with equal number of fraudulent and legitimate data.
    '''
    majority = df[df['Class'] == 0]
    minority = df[df['Class'] != 0]
    majority = majority.sample(n = len(minority))
    df = pd.concat([majority, minority], axis=0)
    return df

# 1.2.2 Oversampling in Practice
def random_oversampling(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Given credit card dataset with 0 as the majority for 'Class' column. Returns credit card data with two classes
    having the same shape, given raw data read from csv.

    Parameters
    ----------
    df: pd.DataFrame
        The potentially imbalanced dataset.

    Returns
    -------
    Oversampled dataset with equal number of fraudulent and legitimate data.
    '''
    majority = df[df['Class'] == 0]
    minority = df[df['Class'] != 0]
    minority = minority.sample(n = len(majority), replace=True)
    df = pd.concat([majority, minority], axis=0)
    return df

# 2.1 Splitting Data
def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float=0.25):
    '''
    Randomly split the data into two sets.

    Parameters
    ----------
    X: np.ndarray
        (m, n) dataset (features).
    y: np.ndarray
        (m,) dataset (corresponding targets).
    test_size: np.float64
        fraction of the dataset to be put in the test dataset.

    Returns
    -------
    A tuple of four elements (X_train, X_test, y_train, y_test):
    X_train: np.ndarray
        (m - k, n) training dataset (features).
    X_test: np.ndarray
        (k, n) testing dataset (features).
    y_train: np.ndarray
        (m - k,) training dataset (corresponding targets).
    y_test: np.ndarray
        (k,) testing dataset (corresponding targets).
    '''

    m = X.shape[0] # number of features
    k = int(m * test_size)

    df = np.column_stack((X, y))
    df = pd.DataFrame(df)

    trainSample = df.sample(frac = 1 - test_size)
    testSample = df.drop(trainSample.index)

    y_train = trainSample[trainSample.columns[-1]]
    X_train = trainSample[trainSample.columns[:-1]] 

    y_test = testSample[testSample.columns[-1]]
    X_test = testSample[testSample.columns[:-1]] 

    return (X_train, X_test, y_train, y_test)

# 3.1 Cost Function
def cost_function(X: np.ndarray, y: np.ndarray, weight_vector: np.ndarray):
    '''
    Cross entropy error for logistic regression

    Parameters
    ----------
    X: np.ndarray
        (m, n) training dataset (features).
    y: np.ndarray
        (m,) training dataset (corresponding targets).
    weight_vector: np.ndarray
        (n,) weight parameters.

    Returns
    -------
    Cost
    '''
    
    # Machine epsilon for numpy `float64` type
    eps = np.finfo(np.float64).eps

    numDataPoints = len(X)
    yPredVector = np.matmul(X, weight_vector)
    ySigmoidVector = 1 / (1 + np.exp(-1 * yPredVector))
    errorVector = -y * np.log(ySigmoidVector + eps) - (1 - y) * np.log(1 - ySigmoidVector + eps)

    return np.sum(errorVector) / numDataPoints

# 3.2 Weight Update
def weight_update(X: np.ndarray, y: np.ndarray, alpha: np.float64, weight_vector: np.ndarray) -> np.ndarray:
    '''
    Do the weight update for one step in gradient descent

    Parameters
    ----------
    X: np.ndarray
        (m, n) training dataset (features).
    y: np.ndarray
        (m,) training dataset (corresponding targets).
    alpha: np.float64
        logistic regression learning rate.
    weight_vector: np.ndarray
        (n,) weight parameters.

    Returns
    -------
    New weight vector after one round of update.
    '''
    n = len(X)

    yPredVector = np.dot(X, weight_vector)
    ySigmoidVector = 1 / (1 + np.exp(-1 * yPredVector))
    diffVector = ySigmoidVector - y
    partialVector = alpha * np.dot(np.transpose(X), diffVector) / n
    weight_vector -= partialVector

    return weight_vector
    
# 3.3 Logistic regression classification
def logistic_regression_classification(X: np.ndarray, weight_vector: np.ndarray, prob_threshold: np.float64=0.5):
    '''
    Do classification task using logistic regression.

    Parameters
    ----------
    X: np.ndarray
        (m, n) training dataset (features).
    weight_vector: np.ndarray
        (n,) weight parameters.
    prob_threshold: np.float64
        the threshold for a prediction to be considered fraudulent.

    Returns
    -------
    Classification result as an (m,) np.ndarray
    '''
    yPredVector = np.matmul(X, weight_vector)
    diffVector = 1 / (1 + np.exp(-1 * yPredVector))
    diffVector[diffVector > prob_threshold] = 1
    diffVector[diffVector <= prob_threshold] = 0

    return diffVector

# 3.4 Logistic Regression Using Batch Descent
def logistic_regression_batch_gradient_descent(X_train: np.ndarray, y_train: np.ndarray, max_num_epochs: int = 250, threshold: np.float64 = 0.05, alpha: np.float64 = 1e-5):
    '''
    Initialize your weight to zeros. Write your terminating condition, and run the weight update for some iterations.
    Get the resulting weight vector. Use that to do predictions, and output the final weight vector.

    Parameters
    ----------
    X_train: np.ndarray
        (m, n) training dataset (features).
    y_train: np.ndarray
        (m,) training dataset (corresponding targets).
    max_num_epochs: int
        this should be one of the terminating conditions. 
        That means if you initialize num_update_rounds to 0, 
        then you should stop updating the weights when num_update_rounds >= max_num_epochs.
    threshold: np.float64
        terminating when error <= threshold value, or if you reach the max number of update rounds first.
    alpha: np.float64
        logistic regression learning rate.

    Returns
    -------
    The final (n,) weight parameters
    '''

    # TODO: add your solution here and remove `raise NotImplementedError`
    raise NotImplementedError



if __name__ == "__main__":
    data1 = [[111.1, 10, 0], [111.2, 20, 0], [111.3, 10, 0], [111.4, 10, 0], [111.5, 10, 0], [211.6, 80, 1],[111.4, 10, 0], [111.5, 80, 1], [211.6, 80, 1]]
    df1 = pd.DataFrame(data1, columns = ['V1', 'V2', 'Class'])
    X1 = df1.iloc[:, :-1].to_numpy()
    y1 = df1.iloc[:, -1].to_numpy()
    w1 = np.transpose([-0.000002, 0.000003])
    expected1 = np.transpose([0, 0, 0, 0, 0, 0, 0, 1, 0])
    result1 = logistic_regression_classification(X1, w1)
    print(result1)
    print(expected1)


    assert result1.shape == expected1.shape and (result1 == expected1).all()
        
