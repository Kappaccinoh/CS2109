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

    weight_vector = 
    weightX_vector = np.multiply(X, weight_vector)


    


if __name__ == "__main__":
    data1 = [[111.1, 10, 0], [111.2, 20, 0], [111.3, 30, 0], [111.4, 40, 0], [111.5, 50, 0], [111.6, 60, 1],
        [111.7, 70, 0], [111.8, 80, 1], [111.9, 90, 1]]
    df1 = pd.DataFrame(data1, columns = ['V1', 'V2', 'Class'])
    X1 = df1.iloc[:, :-1].to_numpy()
    y1 = df1.iloc[:, -1].to_numpy()
    expected1 = [7, 2, 7, 2]

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25)

    assert len(X1_train) == expected1[0] and\
    len(X1_test) == expected1[1] and\
    len(y1_train) == expected1[2] and\
    len(y1_test) == expected1[3]

    
    
