import csv
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def load_rgb_image(image_file_path):
    '''
    Loads the RGB image at `image_file_path`, where the file path should be
    specified relative to this notebook, and returns the image as an `ndarray`
    with shape `(h, w, 3)`, where `h` and `w` are the height and width of the
    image respectively.

    NOTE: every entry in the returned `ndarray` is an integer between 0 and 255,
    inclusive.
    '''

    dirname = os.path.abspath('')
    image_path = os.path.join(dirname, image_file_path)
    image = mpimg.imread(image_path)
    return image

def display_image(image):
    '''
    Displays image that is represented by `image`, an `ndarray`.

    NOTE: if the data type of `image` is `int`, its entries should have values
    between 0 and 255 (inclusive); otherwise, its entries should have values
    between 0 and 1 (inclusive).
    '''
    plt.imshow(image)

def _load_digits_data(is_train):
    '''
    Loads handwritten digits dataset. 

    Parameter
    ---------
    is_train: bool
        If `is_train` is `True`, the dataset returned will be unlabelled; otherwise,
        it is labelled
  
    Returns
    -------
    An `m * n` matrix `samples`. Here, `m` is the number of samples.

    If `is_train` is `True`, then `n` is equal to `h * w`, where `h` denotes the
    image height and `w` denotes the image width.
    '''
    dirname = os.path.abspath('')
    file_name = 'digits_train.csv' if is_train else 'digits_validation.csv'
    file_path = os.path.join(dirname, file_name)
    data = []
  
    with open(file_path, mode='r') as file:
        rows = csv.reader(file)
        for row in rows:  
            data.append([int(num) for num in row])

    return np.array(data)

def load_digits_data_train():
    '''
    Loads the training dataset for the handwritten digits recognition problem.

    Returns
    -------
    A 2D array `digits`, where `digits[i].reshape((28, 28))` is the image of the `i`th 
    handwritten digit. This image only has one channel, i.e. every pixel is
    only represented by an intensity value rather than an RGB triplet.
    '''
    return _load_digits_data(True)

def load_digits_data_validation():
    '''
    Loads the validation dataset for the handwritten digits recognition problem.

    Returns
    -------
    A tuple (`digits`, `labels`). 

    `digits` is a 2D array, where `digits[i].reshape((28, 28))` is the image of 
    the `i`th handwritten digit. This image only has one channel, i.e. every pixel 
    is only represented by an intensity value rather than an RGB triplet.

    `labels` is an array where `labels[i]` returns the actual label of the `i`th
    handwritten digit in `digits`. Note that `labels[i]` is an integer such that
    0 <= `labels[i]` <= 9.
    '''
    data = _load_digits_data(False)
    digits = data[:, 1:]
    labels = data[:, 0]
    return digits, labels

def compute_accuracy(pred_labels, true_labels):
    '''
    Computes the accuracy of the predicted labels, given the true labels.
    '''
    return np.sum(pred_labels == true_labels) / true_labels.shape[0]

# Task 1.1.1
def assign_clusters(X, centroids):
    """
    Assigns each sample in X to the closest cluster.

    Parameters
    ----------
    X: np.darray
        An `m * n` matrix where `m` is the number of samples and `n` is the
        number of features which each sample has. In other words, the `i`th sample
        is given by `X[i]`.
    centroids: np.darray
        An `n_clusters * n` matrix where `n_clusters` is the number of clusters
        and `n` is the number of features which each sample has. In particular, 
        `centroids[j]` represents the `j`th cluster's centroid.

    Returns
    -------
    An `ndarray` of integers that indicates the cluster assignment for each sample.
    Specifically, if `labels` is the `ndarray` returned, then `labels[i]` indicates
    that the `i`th sample in X has been assigned to the `labels[i]`th cluster, where
    `labels[i]` is a value in the interval [0, `n_clusters`). This cluster should
    be the one with a centroid that is closest to `X[i]` in terms of its Euclidean
    distance. Note that this array should be an array of integers.

    Note
    ----
    If there are multiple possible closest clusters for the `i`th sample in X,
    assign it to the cluster with the smallest index. For example, if `X[0]` is
    as close to `centroids[0]` as it is to `centroids[1]`, it should be assigned
    to the 0th cluster instead of the 1st cluster, since 0 < 1.
    """
    # TODO: add your solution here and remove `raise NotImplementedError`
    # at most 1 loop allowed

    # Calculate Euclidean distances between each sample and each centroid
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
    
    # Assign each sample to the cluster with the closest centroid
    labels = np.argmin(distances, axis=1)
    
    return labels

def update_centroids(X, labels, n_clusters):
    '''
    Updates the centroids based on the (new) assignment of clusters.

    Parameters
    ----------
    X: np.darray
        An `m * n` matrix where `m` is the number of samples and `n` is the
        number of features which each sample has. In other words, the `i`th sample
        is given by `X[i]`.
    labels: np.darray
        An array of `m` values, where `m` is the number of samples, that indicates
        which cluster the samples have been assigned to, i.e. the `i`th
        sample is assigned to the `labels[i]`th cluster.
    n_clusters: int
        No. of clusters.

    Returns
    -------
    The `centroids`, an `ndarray` with shape `(n_clusters, n)`, for each cluster,
    based on the current cluster assignment as specified by `labels`. In particular,
    `centroids[j]` returns the centroid for the `j`th cluster.
    '''
    # TODO: add your solution here and remove `raise NotImplementedError`  
    # at most 1 loop allowed
    centroids = np.array([np.mean(X[labels == j], axis=0) for j in range(n_clusters)])
    
    return centroids

def check_convergence(prev_centroids, centroids, threshold):
    '''
    Checks whether the algorithm has converged.

    Parameters
    ----------
    prev_centroids: np.darray
        An `n_clusters * n` matrix where `n_clusters` is the number of clusters
        and `n` is the number of features which each sample has. In particular, 
        `prev_centroids[j]` represents the `j`th cluster's centroid in the
        PREVIOUS iteration.
    centroids: np.darray
        An `n_clusters * n` matrix where `n_clusters` is the number of clusters
        and `n` is the number of features which each sample has. In particular, 
        `centroids[j]` represents the `j`th cluster's centroid in the CURRENT
        iteration.
    threshold: double
        If each cluster is such that the Euclidean distance between its centroids
        in the current and previous iteration is strictly less than `threshold`,
        the algorithm is deemed to have converged.

    Returns
    -------
    `True` if and only if the Euclidean distance between each
    cluster's centroid in the previous and current iteration is strictly
    less than `threshold`.
    '''
    # no loop allowed
    # Calculate Euclidean distances between each pair of centroids
    distances = np.linalg.norm(prev_centroids - centroids, axis=1)
    
    # Check if all distances are strictly less than the threshold
    return np.all(distances < threshold)

if __name__ == "__main__":
    # Public test case 1
    X_111 = np.arange(20).reshape((5, 4))
    labels_111 = assign_clusters(X_111, np.copy(X_111))

    assert np.issubdtype(labels_111.dtype, int)

    # Public test case 2
    assert np.all(labels_111 == np.arange(5))
