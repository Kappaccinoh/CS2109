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

    shuffled_indices = np.random.permutation(m)
    X_labels_shuffled = X[shuffled_indices]
    y_labels_shuffled = y[shuffled_indices]

    y_train = y_labels_shuffled[: m - k]
    X_train = X_labels_shuffled[: m - k]

    y_test = y_labels_shuffled[m - k : m]
    X_test = X_labels_shuffled[m - k : m] 


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
    temp = -1 * yPredVector
    ySigmoidVector = 1 / (1 + np.exp(temp))
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

    # yPredVector = np.dot(X, weight_vector)
    # ySigmoidVector = 1 / (1 + np.exp(-1 * yPredVector))
    # diffVector = ySigmoidVector - y
    # partialVector = alpha * np.dot(np.transpose(X), diffVector) / n
    # partialVector = np.transpose(partialVector)
    # weight_vector = weight_vector - partialVector
    # print(weight_vector.shape)

    partial0 = np.sum(np.matmul(X, weight_vector), axis=0)
    partial0 = 1 / (1 + np.exp(-1 * partial0))
    partial0 = partial0 - y
    partial1 = np.matmul(np.array(X).transpose(), partial0) / n
    weight_vector = weight_vector - alpha * partial1

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
    epoch = 0
    weight_vector = np.array([0])
    weight_vector = np.repeat(weight_vector, len(X_train[0]))
    error = cost_function(X_train, y_train, weight_vector)

    while (True):
        weight_vector = weight_update(X_train, y_train, alpha, weight_vector)
        error = cost_function(X_train, y_train, weight_vector)
        epoch += 1
        if (epoch >= max_num_epochs or error < threshold):
            break
        
    return weight_vector

# Task 3.5 Logistic Regression Using Stochastic Gradient Descent
def weight_update_stochastic(X: np.ndarray, y: np.ndarray, alpha: np.float64, weight_vector: np.ndarray) -> np.ndarray:
    '''
    Do the weight update for one step in gradient descent.

    Parameters
    ----------
    X: np.ndarray
        (1, n) training dataset (features).
    y: np.ndarray
        one y in training dataset (corresponding targets).
    alpha: np.float64
        logistic regression learning rate.
    weight_vector: np.ndarray
        (n, 1) vector of weight parameters.

    Returns
    -------
    New (n,) weight parameters after one round of update.
    '''
    n = X.shape[1]

    yPredVector = np.dot(X, weight_vector)
    ySigmoidVector = 1 / (1 + np.exp(-1 * yPredVector))
    diffVector = ySigmoidVector - y
    partialVector = alpha * np.dot(np.transpose(X), diffVector) / n

    weight_vector = weight_vector - partialVector
    return weight_vector

    # partial0 = np.sum(np.matmul(X, weight_vector), axis=0)
    # partial0 = 1 / (1 + np.exp(-1 * partial0))
    # partial0 = partial0 - y
    # partial1 = np.matmul(np.array(X).reshape(1,n).transpose(), partial0) / n
    # weight_vector = weight_vector - alpha * partial1

    # return weight_vector


# Task 3.5 Continued
def logistic_regression_stochastic_gradient_descent(X_train: np.ndarray, y_train: np.ndarray, max_num_iterations: int=250, threshold: np.float64=0.05, alpha: np.float64=1e-5) -> np.ndarray:
    '''
    Initialize your weight to zeros. Write a terminating condition, and run the weight update for some iterations.
    Get the resulting weight vector.

    Parameters
    ----------
    X_train: np.ndarray
        (m, n) training dataset (features).
    y_train: np.ndarray
        (m,) training dataset (corresponding targets).
    max_num_iterations: int
        this should be one of the terminating conditions. 
        The gradient descent step should happen at most max_num_iterations times.
    threshold: np.float64
        terminating when error <= threshold value, or if you reach the max number of update rounds first.
    alpha: np.float64
        logistic regression learning rate.

    Returns
    -------
    The final (n,) weight parameters
    '''
    m = X_train.shape[0]
    n = X_train.shape[1]
    epoch = 0
    weight_vector = np.zeros(n)
    error = cost_function(X_train, y_train, weight_vector)

    while (True):
        if (epoch >= max_num_iterations or error <= threshold):
            break
        
        # Select one random entry from X_train to pass on to weight_update_stochastic
        randChoice = np.random.randint(0, m)
        tempX = X_train[randChoice].reshape([1, n])
        tempY = y_train[randChoice]
        weight_vector = weight_update_stochastic(tempX, tempY, alpha, weight_vector)
        error = cost_function(X_train, y_train, weight_vector)
        epoch += 1
    
    return weight_vector

# Task 3.6 Graph Plotting
import matplotlib.pyplot as plt
from time import time

# Plot 1 - Plot of cross entropy loss against number of update rounds

# dirname = os.getcwd()
# credit_card_data_filepath = os.path.join(dirname, 'credit_card_medium.csv')

# credit_df = pd.read_csv(credit_card_data_filepath)
# X = credit_df.values[:, :-1]
# y = credit_df.values[:, -1:]

# X_sample, y_sample = X, y
# num_interations = 2000
# batch_rounds = []
# batch_costs = []

# for i in range(50, num_interations + 1, 50):
#     weight_vector = logistic_regression_batch_gradient_descent(X_sample, y_sample, i, 0, 1e-5)
#     batch_rounds.append(i)
#     batch_costs.append(cost_function(X_sample, y_sample, weight_vector))
# plt.plot(batch_rounds, batch_costs, 'g^', label="Batch Gradient Descent")

# stochastic_rounds = []
# stochastic_costs = []
# for i in range(50, num_interations + 1, 50):
#     weight_vector = logistic_regression_stochastic_gradient_descent(X_sample, y_sample, i, 0, 1e-5)
#     stochastic_rounds.append(i)
#     stochastic_costs.append(cost_function(X_sample, y_sample, weight_vector))
# plt.plot(stochastic_rounds, stochastic_costs, 'bs', label="Stochastic Gradient Descent")

# plt.xlabel('Number of update rounds')
# plt.ylabel('Cross Entropy Loss')
# plt.legend()
# plt.title('Plot of cross entropy loss against number of update rounds')

# plt.show()



# Plot 2 - Plot of cross entropy loss against runtime (sec)

# dirname = os.getcwd()
# credit_card_data_filepath = os.path.join(dirname, 'credit_card_medium.csv')

# credit_df = pd.read_csv(credit_card_data_filepath)
# X = credit_df.values[:, :-1]
# y = credit_df.values[:, -1:]

# X_sample, y_sample = X, y
# num_interations = 2000
# batch_times = []
# batch_costs = []

# for i in range(50, num_interations + 1, 50):
#     start = time()
#     weight_vector = logistic_regression_batch_gradient_descent(X_sample, y_sample, i, 0, 1e-5)
#     stop = time()
#     batch_times.append(stop - start)
#     batch_costs.append(cost_function(X_sample, y_sample, weight_vector))
# plt.plot(batch_times, batch_costs, 'g^', label="Batch Gradient Descent")

# stochastic_times = []
# stochastic_costs = []
# for i in range(50, num_interations + 1, 50):
#     start = time()
#     weight_vector = logistic_regression_stochastic_gradient_descent(X_sample, y_sample, i, 0, 1e-5)
#     stop = time()
#     stochastic_times.append(stop - start)
#     stochastic_costs.append(cost_function(X_sample, y_sample, weight_vector))
# plt.plot(stochastic_times, stochastic_costs, 'bs', label="Stochastic Gradient Descent")

# plt.xlabel('Runtime (sec)')
# plt.ylabel('Cross Entropy Loss')
# plt.legend()
# plt.title('Plot of cross entropy loss against runtime (sec)')

# plt.show()

# Task 4.1
def multi_class_logistic_regression_batch_gradient_descent(X_train: np.ndarray, y_train: np.ndarray, max_num_epochs: int, threshold: np.float64, alpha: np.float64, class_i: str) -> np.ndarray:
    '''
    Initialize your weight to zeros. Write your terminating condition, and run the weight update for some iterations.
    Get the resulting weight vector. Output the resulting weight vector.

    Parameters
    ----------
    X_train: np.ndarray
        (m, n) training dataset (features).
    y_train: np.ndarray
        (m,) training dataset (corresponding targets).
    alpha: np.float64
        logistic regression learning rate.
    max_num_epochs: int
        this should be one of the terminating conditions. 
        That means if you initialize num_update_rounds to 0, 
        then you should stop updating the weights when num_update_rounds >= max_num_epochs.
    threshold: float
        terminating when error <= threshold value, or if you reach the max number of update rounds first.
    class_i: string
        one of 'none', 'full', 'some'.

    Returns
    -------
    The final (n,) weight parameters
    '''
    # df = pd.DataFrame(np.concatenate((X_train, y_train), axis=1), columns=['max_capcity', 'feedback_score', 'average_expense', 'occupancy'])
    # df.loc[df['occupancy'] == class_i] = 1.0
    # df.loc[df['occupancy'] != 1.0] = 0.0

    # X_train_New = df.values[:, :-1]
    # y_train_New = df.values[:, -1:]
    m = X_train.shape[0]
    n = X_train.shape[1]

    y_train = y_train.reshape(m,1)
    combinedArray = np.hstack((X_train, y_train))
    m = len(combinedArray)
    n = len(combinedArray[0])
    combinedArray[:, -1:][combinedArray[:, -1:] != class_i] = 0
    combinedArray[:, -1:][combinedArray[:, -1:] == class_i] = 1
    X_train_New = combinedArray[:, :-1]
    y_train_New = combinedArray[:, -1:]
    y_train_New = y_train_New.flatten()

    X_train_New = np.asarray(X_train_New, float)
    y_train_New = np.asarray(y_train_New, float)

    return logistic_regression_batch_gradient_descent(X_train_New, y_train_New, max_num_epochs, threshold, alpha)

# Task 4.2 Multi-class classification
def multi_class_logistic_regression_classification(X: np.ndarray, weight_vector_none: np.ndarray, weight_vector_some: np.ndarray, weight_vector_full: np.ndarray):
    '''
    Do classification task using logistic regression.
    In the case of a tie, break the tie in the priority 'none' > 'some' > 'full'.

    Parameters
    ----------
    X: np.ndarray
        (m, n) training dataset (features)
    weight_vector_none: np.ndarray
        (n,) weight parameters for the 'none' class.
    weight_vector_some: np.ndarray
        (n,) weight parameters for the 'some' class.
    weight_vector_full: np.ndarray
        (n,) weight parameters for the 'full' class.

    Returns
    -------
    Classification result as an (m,) np.ndarray. Eg ['some', 'none', 'full', ... ,'none'].
    '''
    m = X.shape[0]
    n = X.shape[1]

    yPredVector_none = np.matmul(X, weight_vector_none)
    diffVector_none = 1 / (1 + np.exp(-1 * yPredVector_none))

    yPredVector_some = np.matmul(X, weight_vector_some)
    diffVector_some = 1 / (1 + np.exp(-1 * yPredVector_some))

    yPredVector_full = np.matmul(X, weight_vector_full)
    diffVector_full = 1 / (1 + np.exp(-1 * yPredVector_full))

    diffVector_none = diffVector_none.reshape(m,1)
    diffVector_some = diffVector_some.reshape(m,1)
    diffVector_full = diffVector_full.reshape(m,1)

    combinedVector = np.hstack((diffVector_none, diffVector_some))
    combinedVector = np.hstack((combinedVector, diffVector_full))

    max_indices = np.argmax(combinedVector, axis=1).astype('object')
    max_indices[max_indices == 0] = "none"
    max_indices[max_indices == 1] = "some"
    max_indices[max_indices == 2] = "full"

    return max_indices

from sklearn import svm
from sklearn import model_selection
from sklearn.metrics import accuracy_score

# Task 5.1 
def linear_svm(X: np.ndarray, y: np.ndarray):
    '''
    Do classification using linear svm. Given X and y, note that here X and y are not training sets, but rather the
    entire dataset. Do a train test data split with test_size=0.3, and random_state=42.

    Parameters
    ----------
    X: np.ndarray
        (m, n) whole dataset (features)
    y: np.ndarray
        (m,) whole dataset (corresponding targets)

    Returns
    -------
    pred: np.ndarray
        The predictions.
    acc: np.float64
        The accuracy on a scale up to 100.
    '''
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)

    svm_classifier = svm.SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    predictions = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100
    return predictions, accuracy

# Task 5.2
def gaussian_kernel_svm(X: np.ndarray, y: np.ndarray):
    '''
    Do classification using Gaussian Kernel svm. Given X and y, note that here X and y are not training sets, but
    rather the entire dataset. Do a train test data split with test_size=0.3, and random_state=42.

    Parameters
    ----------
    X: np.ndarray
        (m, n) whole dataset (features)
    y: np.ndarray
        (m,) whole dataset (corresponding targets)

    Returns
    -------
    pred: np.ndarray
        The predictions.
    acc: np.float64
        The accuracy on a scale up to 100.
    '''
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)

    svm_classifier = svm.SVC(kernel='rbf')
    svm_classifier.fit(X_train, y_train)
    predictions = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100
    return predictions, accuracy

if __name__ == "__main__":
    # small data
    data1 = [[111.1, 10, -1], [111.2, 20, -1], [111.3, 10, -1], [111.4, 10, -1], [111.5, 10, -1], [211.6, 80, 1],
            [111.4, 10, -1], [111.5, 80, 1], [211.6, 80, 1]]
    df1 = pd.DataFrame(data1, columns = ['V1', 'V2', 'Class'])
    X1 = df1.iloc[:, :-1].to_numpy()
    y1 = df1.iloc[:, -1].to_numpy()
    expected1_y = np.transpose([-1, -1, 1])
    expected1_accuracy = 66.66666666666666
    result1 = gaussian_kernel_svm(X1, y1)
    assert (result1[0] == expected1_y).all() and (result1[0]).shape == expected1_y.shape and round(result1[1], 5) == round(expected1_accuracy, 5)


    # subset of credit card data
    class_0 = credit_df[credit_df['Class'] == 0]
    class_1 = credit_df[credit_df['Class'] == 1]

    data_0 = class_0.sample(n=15, random_state=42)
    data_1 = class_1.sample(n=50, random_state=42)
    data_100 = pd.concat([data_1, data_0], axis=0)
    X = data_100.iloc[:, :-1].to_numpy()
    y = data_100.iloc[:, -1].to_numpy()

    expected_pred = np.transpose([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    expected_accuracy = 80.0
    result = gaussian_kernel_svm(X, y.ravel())
    assert (result[0] == expected_pred).all() and (result[0]).shape == expected_pred.shape and round(result[1], 5) == round(expected_accuracy, 5)