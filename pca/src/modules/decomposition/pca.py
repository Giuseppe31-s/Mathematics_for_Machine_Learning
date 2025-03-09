from sklearn.preprocessing import StandardScaler
import numpy as np




# GRADED FUNCTION: DO NOT EDIT THIS LINE
def normalize(X):
    """Normalize the given dataset X to have zero mean.
    Args:
        X: ndarray, dataset of shape (N,D) where D is the dimension of the data,
           and N is the number of datapoints
    
    Returns:
        (Xbar, mean): tuple of ndarray, Xbar is the normalized dataset
        with mean 0; mean is the sample mean of the dataset.
    """
    mu = np.mean(X, axis = 0) 
    Xbar = X - mu       
    return Xbar, mu


def eig(S):
    """Compute the eigenvalues and corresponding eigenvectors
        for the covariance matrix S.
    Args:
        S: ndarray, covariance matrix

    Returns:
        (eigvals, eigvecs): ndarray, the eigenvalues and eigenvectors

    Note:
        the eigenvals and eigenvecs should be sorted in descending
        order of the eigen values
    """
    # YOUR CODE HERE
    # Uncomment and modify the code below
    # Compute the eigenvalues and eigenvectors
    # Note that you can compute both of these with just a single function call
    eigvals, eigvecs = np.linalg.eig(S)

    # The eigenvalues and eigenvectors need to be sorted in descending order according to the eigenvalues
    # We will use `np.argsort` to find a permutation of the indices of eigvals that will sort eigvals in ascending order and
    # then find the descending order via [::-1], which reverse the indices
    # (https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html)
    sort_indices = np.argsort(eigvals)[::-1]

    # Notice that we are sorting the columns (not rows) of eigvecs since the columns represent the eigenvectors.
    return eigvals[sort_indices], eigvecs[:, sort_indices]


# GRADED FUNCTION: DO NOT EDIT THIS LINE
def projection_matrix(B):
    """Compute the projection matrix onto the space spanned by `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace

    Returns:
        P: the projection matrix
    """
    # YOUR CODE HERE

    # Uncomment and modify the code below
    return B @ np.linalg.inv(B.T @ B) @ B.T


# GRADED FUNCTION: DO NOT EDIT THIS LINE
def PCA(X, num_components):
    """
    Args:
        X: ndarray of size (N, D), where D is the dimension of the data,
           and N is the number of datapoints
        num_components: the number of principal components to use.
    Returns:
        the reconstructed data, the sample mean of the X, principal values
        and principal components
    """

    # YOUR CODE HERE
    # your solution should take advantage of the functions you have implemented above.
    ### Uncomment and modify the code below
    # first perform normalization on the digits so that they have zero mean and unit variance
  
    X_normalized, mean = normalize(X)
    # Then compute the data covariance matrix S
    S = X_normalized.T @ X_normalized * 1 / X_normalized.shape[0]

    # Next find eigenvalues and corresponding eigenvectors for S
    eig_vals, eig_vecs = eig(S)
    # Take the top `num_components` of eig_vals and eig_vecs,
    # This will be the corresponding principal values and components
    # Remember that the eigenvectors are the columns of the matrix `eig_vecs`
    principal_vals, principal_components = (
        eig_vals[:num_components],
        eig_vecs[:, :num_components],
    )

    # Due to precision errors, the eigenvectors might come out to be complex, so only take their real parts
    principal_components = np.real(principal_components)
    principal_vals = np.real(principal_vals)

    # Reconstruct the data by projecting the normalized data on the basis spanned by the principal components
    # Remember that the data points in X_normalized are arranged along the rows
    # but while projecting, we need them to be arranged along the columns
    # Notice that we have subtracted the mean from X so make sure that you add it back
    # to the reconstructed data
    reconst = ((projection_matrix(principal_components) @ X_normalized.T).T) + mean
    return reconst, principal_vals, principal_components, mean
