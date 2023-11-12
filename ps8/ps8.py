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

def k_means_once(X, initial_centroids, threshold):
    '''
    Assigns each point in X to a cluster by running the K-Means algorithm
    once till convergence.

    Parameters
    ----------
    X: np.darray
        An `m * n` matrix where `m` is the number of samples and `n` is the
        number of features which each sample has. In other words, the `i`th sample
        is given by `X[i]`.
    initial_centroids: np.darray
        An `n_clusters * n` matrix, where `n_clusters` is the number of clusters and
        `n` is the number of features that each sample in X has. This matrix is such
        that the `i`th row represents the initial centroid of the `i`th cluster.
    threshold: double
        During the clustering process, if the difference in centroids between
        two consecutive iterations is less than `threshold`, the algorithm is
        deemed to have converged.

    Returns
    -------
    The cluster assignment for each sample, and the `n_clusters` centroids found. 
    In particular, the cluster assignment for the ith sample in `X` is given by `labels[i]`,
    where 0 <= `labels[i]` < `n_clusters`. Moreover, suppose c = `labels[i]`. Then,
    the `i`th sample belongs to the cluster c with the centroid given by `centroids[c]`.
    '''
    # TODO: add your solution here and remove `raise NotImplementedError`
    # at most 1 loop allowed
    # Initialize centroids to the initial centroids
    centroids = np.copy(initial_centroids)
    
    # Initialize previous centroids to zeros
    prev_centroids = np.zeros_like(centroids)
    
    # Iterate until convergence
    while not check_convergence(prev_centroids, centroids, threshold):
        # Assign each sample to the closest cluster
        labels = assign_clusters(X, centroids)
        
        # Update centroids based on the new assignment
        prev_centroids = np.copy(centroids)
        centroids = update_centroids(X, labels, initial_centroids.shape[0])
    
    return labels, centroids

def compute_loss(X, centroids, labels):
    '''
    Computes the loss based on the current assignment of clusters.

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
    labels: np.darray
        An array of `m` values, where `m` is the number of samples, that indicates
        which cluster the samples have been assigned to,  i.e. `labels[i]` indicates
        that the `i`th sample is assigned to the `labels[i]`th cluster.
  
    Returns
    -------
    The loss based on the current assignment of clusters.
    '''
    # TODO: add your solution here and remove `raise NotImplementedError`
    # no loop allowed
    # Calculate the squared Euclidean distances between each sample and its assigned centroid
    distances_squared = np.linalg.norm(X - centroids[labels], axis=1)**2
    
    # Compute the average loss
    loss = np.mean(distances_squared)
    
    return loss

def init_centroids(X, n_clusters, random_state):
    '''
    Initialises the centroids that will be used for K-Means, by randomly
    picking `n_clusters` points from `X` and using these points as the 
    initial centroids.

    Parameters
    ----------
    X: np.darray
        An `m * n` matrix where `m` is the number of samples and `n` is the
        number of features which each sample has. In other words, the `i`th sample
        is given by `X[i]`.
    n_clusters: int
        No. of clusters.
    random_state: int or `None`
        Used to make the algorithm deterministic, if specified.

    Returns
    -------
    An `ndarray` with the shape `(n_clusters, n)` such that the `i`th row
    represents the `i`th randomly chosen centroid.
    '''
    # no loop allowed
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    sample_indices = rng.permutation(n_samples)[:n_clusters]
    return X[sample_indices]

import numpy as np

def k_means(X, n_clusters, threshold, n_init=1, random_state=None):
    '''
    Clusters samples in X using the K-Means algorithm.

    Parameters
    ----------
    X: np.darray
        An `m * n` matrix where `m` is the number of samples and `n` is the
        number of features which each sample has. In other words, the `i`th sample
        is given by `X[i]`.
    n_clusters: int
        No. of clusters.
    threshold: float
        Threshold that determines when the algorithm should terminate. If between
        two consecutive iterations the cluster centroids' difference is less than
        `threshold`, terminate the algorithm, i.e. suppose `c_i` is the ith
        centroid in the kth iteration, and `c'_i` is the ith centroid in the
        (k + 1)th iteration, we terminate the algorithm if and only if for all  
        i, d(`c_i`, `c'_i`) < `threshold`, where d is the distance function.
    n_init: int
        No. of times to run K-means.
    random_state: int or `None`
        Used to make the algorithm deterministic, if specified.
  
    Returns
    -------
    The cluster assignment for each sample, and the `n_clusters` centroids found. 
    In particular, the cluster assignment for the ith sample in `X` is given by `labels[i]`,
    where 0 <= `labels[i]` < `n_clusters`. Moreover, suppose c = `labels[i]`. Then,
    the `i`th sample belongs to the cluster c with the centroid given by `centroids[c]`.

    If `n_init` > 1, then the labels and corresponding centroids that result in
    the lowest distortion will be returned.

    Note
    ----
    If `n_init` is greater than 1, the labels and centroids found from the run
    (out of `n_init` runs) which gives the lowest distortion will be used.
    '''
    best_centroids, best_labels = None, None
    lowest_loss = np.inf

    for i in range(n_init):
        curr_random_state = None if random_state is None else random_state + i
        initial_centroids = init_centroids(X, n_clusters, curr_random_state)
        
        # Run K-Means
        labels, centroids = k_means_once(X, initial_centroids, threshold)
        
        # Compute loss
        loss = compute_loss(X, centroids, labels)
        
        # Update if the current solution has lower loss
        if loss < lowest_loss:
            lowest_loss = loss
            best_centroids = centroids
            best_labels = labels
  
    return best_labels, best_centroids

IMAGE_FILE_PATH = 'images/teddy_bear.jpg'
display_image(load_rgb_image(IMAGE_FILE_PATH))

def compress_image(image, n_colours, threshold, n_init=1, random_state=None):
    '''
    Compresses the given image by reducing the number of colours in the image to
    `n_colours`. The `n_colours` colours should be selected using `k_means`.

    Parameters
    ----------
    image: np.darray
        The image to be compressed. It should be an `h * w * 3` array, where `h` and
        `w` are its height and width of, with integer entries.
    n_colours: int
        No. of colours that the compressed image should contain.
    threshold: double
        A positive numerical value that determines the termination condition of the
        K-Means algorithm. You MUST call `k_means` with this threshold.
    n_init: int
        No. of times to run the K-Means algorithm before the best solution is
        picked and used for compression.
    random_state: int or `None`
        Used to make the algorithm deterministic, if specified. You MUST call
        `k_means` with `random_state` to ensure reproducility.

    Returns
    -------
    An `ndarray` with the shape `(h, w, 3)`, representing the compressed image
    which only contains `n_colours` colours. Note that the entries should be 
    integers, not doubles or floats.
    '''
    # Flatten the image to be compatible with k_means
    h, w, _ = image.shape
    flattened_image = image.reshape((h * w, 3))

    # Run K-Means to get cluster assignments and centroids
    labels, centroids = k_means(flattened_image, n_colours, threshold, n_init, random_state)

    # Recolor each pixel with the centroid of its cluster
    compressed_image = centroids[labels].reshape((h, w, 3))

    return compressed_image.astype(int)

def predict_labels_kmeans(centroids, cluster_to_digit, digits):
    '''
    Predicts the digit labels for each digit in `digits`.

    Parameters
    ----------
    centroids: np.darray
        The centroids of the clusters. Specifically, `centroids[j]` should represent
        the `j`th cluster's centroid.
    cluster_to_digit: np.darray
        A 1D array such that `cluster_to_digit[j]` indicates which digit the `j`th
        cluster represents. For example, if the 5th cluster represents the digit 0,
        then `cluster_to_digit[5]` should evaluate to 0.
    digits: np.darray
        An `m * n` matrix, where `m` is the number of handwritten digits and `n` is
        equal to 28*28. In particular, `digits[i]` represents the image
        of the `i`th handwritten digit that is in the data set.
  
    Returns
    -------
    A 1D np.darray `pred_labels` with `m` entries such that `pred_labels[i]`
    returns the predicted digit label for the image that is represented by
    `digits[i]`.
    '''
    # Step 1: Assign each digit image to a cluster
    cluster_assignments = assign_clusters(digits, centroids)

    # Step 2: Map cluster assignments to digit labels using cluster_to_digit
    pred_labels = np.vectorize(lambda cluster: cluster_to_digit[cluster])(cluster_assignments)

    return pred_labels

def my_pca(X, n_components):
    '''
    Performs PCA on X to reduce it to `n_components`, using the method
    described in lecture but with the 'centering' of X before SVD is done.

    Parameters
    ----------
    X: np.darray
        An `m * n` matrix where `m` is the number of samples and `n` is the
        number of features which each sample has. In other words, the `i`th sample
        is given by `X[i]`.
    n_components: int
        No. of components that the reduced space has.
  
    Returns
    -------
    The tuple `(components, singular_values)`. Here, `components` is an
    `n_components * n` matrix such that `components[i]` returns the `i`th
    principal axis that has the `i`th largest singular value. In addition,
    `singular_values` is a 1D numpy array with `n_components` entries such that
    `singular_values[i]` returns the `i`th singular value.

    Note
    ----
    'centering' here refers to subtracting the mean from X such that the resulting
    X' has a mean of 0 for each feature.
    '''
    # TODO: add your solution here and remove `raise NotImplementedError`
    # no loop allowed
    # Centering the data
    X_centered = X - np.mean(X, axis=0)

    # Calculating the covariance matrix
    covariance_matrix = (1 / X.shape[0]) * np.dot(X_centered.T, X_centered)

    # Performing singular value decomposition
    _, S, Vt = np.linalg.svd(covariance_matrix)

    # Extracting the top n_components principal components
    components = Vt[:n_components]

    # Calculating singular values
    singular_values = S[:n_components]

    return components, singular_values

def find_kmeans_clusters_w_pca(digits, n_categories, threshold=2, n_init=5, random_state=2109, n_components=70):
    """
    Finds the centroids of the `n_categories` clusters given `digits` when PCA
    is used to reduce the dimensionality of each image.

    Parameters
    ----------
    digits: np.ndarray
        An `m * n` matrix, where `m` is the number of handwritten digits and `n` is
        equal to 28*28. In particular, `digits[i]` represents the image of the `i`th
        handwritten digit.
    n_categories: int
        The number of distinct digits.
    threshold: double
        Threshold that determines when the K-means algorithm should terminate. This
        should be used with `k_means`.
    n_init: int
        The number of times to run the K-means algorithm before picking the best
        cluster. This should be used with `k_means`.
    random_state: int or `None`
        Used to make the K-means and PCA deterministic, if specified.
    n_components: int
        The dimension to which each sample point is reduced, using PCA.

    Returns
    -------
    centroids: np.ndarray
        An `n_categories * n_components` matrix, where `centroids[j]` is
        the centroid of the `j`th cluster.
    pca_model: sklearn.decomposition.PCA
        The PCA model that is used to reduce the dimension of each image.
    """
    # Initialize PCA with the specified number of components and random state
    pca = PCA(n_components=n_components, random_state=random_state)
    
    # Fit and transform the data using PCA
    digits_pca = pca.fit_transform(digits)
    
    # Use K-Means clustering on the transformed data
    best_labels, best_centroids = k_means(digits_pca, n_clusters=n_categories, threshold=threshold, n_init=n_init, random_state=random_state)
    
    return best_centroids, pca

def predict_labels_kmeans_w_pca(pca, centroids, cluster_to_digit, digits):
    '''
    Predicts the digit labels for each digit in `digits`.
    
    Parameters
    ----------
    pca: PCA
        The PCA model that is used when training the K-Means clustering model,
        which produced `centroids`.
    centroids: np.darray
        The centroids of the clusters. Specifically, `centroids[j]` should represent
        the `j`th cluster's centroid.
    cluster_to_digit: np.darray
        A 1D array such that `cluster_to_digit[j]` indicates which digit the `j`th
        cluster represents. For example, if the 5th cluster represents the digit 0,
        then `cluster_to_digit[5]` should evaluate to 0.
        digits: np.darray
        An `m * n` matrix, where `m` is the number of handwritten digits and `n` is
        equal to 28*28. In particular, `digits[i]` represents the image
        of the `i`th handwritten digit that is in the data set.
    digits: np.darray
        An `m * n` matrix, where `m` is the number of handwritten digits and `n` is
        equal to 28*28. In particular, `digits[i]` represents the image
        of the `i`th handwritten digit that is in the data set.

    Returns
    -------
    A 1D np.darray `pred_labels` with `m` entries such that `pred_labels[i]`
    returns the predicted digit label for the image that is represented by
    `digits[i]`.
    '''
    # TODO: add your solution here and remove `raise NotImplementedError`
    # no loop allowed
    # Transform input digits using the same PCA model used during training
    digits_pca = pca.transform(digits)
    
    # Assign each digit to the nearest cluster
    cluster_assignments = np.argmin(np.linalg.norm(digits_pca[:, np.newaxis] - centroids, axis=2), axis=1)
    
    # Map cluster labels to digit labels
    pred_labels = np.vectorize(lambda x: cluster_to_digit[x])(cluster_assignments)
    
    return pred_labels

if __name__ == "__main__":
    train_data = load_digits_data_train()
    validation_data = load_digits_data_validation()

    train_digits = train_data
    validation_digits = validation_data[0]
    validation_labels = validation_data[1]