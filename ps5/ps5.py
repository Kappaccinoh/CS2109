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

    total = pd.concat([X, y], axis=0)



if __name__ == "__main__":
    dirname = os.getcwd()
    credit_card_data_filepath = os.path.join(dirname, 'credit_card_small.csv')
    restaurant_data_filepath = os.path.join(dirname, 'restaurant_data.csv')

    credit_df = pd.read_csv(credit_card_data_filepath)
    credit_df = np.array(credit_df)

    X = credit_df[:, :-1]
    y = credit_df[:, -1:]
    print(X)

    res = pd.concat([X, y], axis=0)

    print(res)
    
