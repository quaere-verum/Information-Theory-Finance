import numpy as np

def discrete_crossentropy(p: np.ndarray, q: np.ndarray | None) -> float | np.ndarray:
    """
    Compute the cross-entropy between two categorical distributions p and q.
    Args:
        p (np.ndarray): a categorical distribution of shape (N,) or a collection of categorical
            distributions of shape (M, N)
        q (np.ndarray, optional): a categorical distribution of shape (N,) or a collection of categorical
            distributions of shape (M, N). If None, then p = q
    Returns:
        np.ndarray: the cross-entropy between p and q of shape None or (M,)
    """
    if q is None:
        q = p
    return -np.sum(p * np.log(q), axis=-1)

def discrete_entropy(p: np.ndarray) -> float | np.ndarray:
    """
    Compute the entropy of a categorical distribution p.
    Args:
        p (np.ndarray): a categorical distribution of shape (N,) or a collection of categorical
            distributions of shape (M, N)
    Returns:
        float | np.ndarray: the entropy of p of shape None or (M,)
    """
    return discrete_crossentropy(p, None)

def discrete_kl_divergence(p: np.ndarray, q: np.ndarray) -> float | np.ndarray:
    """
    Compute the Kullback-Leibler divergence between two categorical distributions p and q.
    Args:
        p (np.ndarray): a categorical distribution of shape (N,) or a collection of categorical 
            distributions of shape (M, N)
        q (np.ndarray): a categorical distribution of shape (N,) or a collection of categorical
            distributions of shape (M, N)
    Returns:
        float | np.ndarray: the Kullback-Leibler divergence between p and q of shape None or (M,)
    """
    assert p.shape == q.shape
    p_clipped = np.clip(p, 1e-8, np.inf)
    q_clipped = np.clip(q, 1e-8, np.inf)
    return -np.sum(p * np.log(q_clipped / p_clipped), axis=-1)

def discrete_mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the mutual information between two discrete random variables X and Y from arrays x and y 
    containing realisations of X and Y respectively.
    Args:
        x (np.ndarray): realisations of X of shape (N,)
        y (np.ndarray): realisations of Y of shape (N,)
    Returns:
        float: the empirical mutual information between X and Y
    """
    assert np.issubdtype(x.dtype, np.integer) and np.issubdtype(y.dtype, np.integer)
    x_dist = discrete_distribution(x)
    y_dist = discrete_distribution(y)
    xy_dist = discrete_joint_distribution(x, y).reshape(-1)
    independent_dist = (x_dist.reshape(-1, 1) * y_dist).reshape(-1)
    return discrete_kl_divergence(xy_dist, independent_dist)

def discrete_distribution(x: np.ndarray) -> np.ndarray:
    """
    Compute the (empirical) categorical distribution of the random variable X from a vector of
    realisations of X, stored in the array x. It is assumed that x contains labels of integer
    type, and that the labels are consecutive integers (not necessarily starting at 0).
    Args:
        x (np.ndarray): realisations of X of shape (N,)
    Returns:
        np.ndarray: the empirical categorical distribution of X of shape (m,) where m is the number
        of labels in x
    """
    x = np.asarray(x, dtype=np.int32) - np.min(x)
    x_hist, _ = np.histogram(x, bins=np.max(x) + 1)
    x_dist = x_hist / len(x)
    return x_dist

def discrete_joint_distribution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the (empirical) joint distribution of (X, Y) given realisations of the random variables X and Y, stored
    in the arrays x and y. It is assumed that x and y contain labels of integer type, and that the labels are
    consecutive integers (not necessarily starting at 0).
    Args:
        x (np.ndarray): realisations of X of shape (N,)
        y (np.ndarray): realisations of Y of shape (N,)
    Returns:
        np.ndarray: the empirical joint distribution of (x, y) of shape (m, n), where m is the number of labels in x
        and n is the number of labels in y
    """
    assert x.shape == y.shape
    x = np.asarray(x, dtype=np.int32) - np.min(x)
    y = np.asarray(y, dtype=np.int32) - np.min(y)
    xy_hist, _, _ = np.histogram2d(x, y, bins=(np.max(x) + 1, np.max(y) + 1))
    xy_dist = xy_hist / np.sum(xy_hist)
    return xy_dist


def continuous_binned_mutual_information(x: np.ndarray, y: np.ndarray, x_quantiles: np.ndarray, y_quantiles: np.ndarray) -> float:
    """
    Compute the mutual information between two real-valued random variables X and Y from arrays x and y containing realisations of
    X and Y respectively. First x and y are binned to transform the distributions into a categorical distribution, based on the 
    specified quantiles. Then the mutual information between the categorical distributions is computed.
    Args:
        x (np.ndarray): realisations of X of shape (N,)
        y (np.ndarray): realisations of Y of shape (N,)
        x_quantiles (np.ndarray): quantiles according to which x will be discretised of shape (M,)
        y_quantiles (np.ndarray): quantiles according to which y will be discretised of shape (M,)
    Returns:
        float: the mutual information between the discretised X and Y
    """
    x_bins = np.quantile(x, q=x_quantiles)
    y_bins = np.quantile(y, q=y_quantiles)
    x_binned = np.digitize(x, bins=x_bins)
    y_binned = np.digitize(y, bins=y_bins)
    return discrete_mutual_information(x_binned, y_binned)


if __name__ == '__main__':
    x = np.random.choice(2, 100)
    y = x + 1
    print(f"Deterministic mutual information: {discrete_mutual_information(x, y):.3f}")
    y = np.random.choice(2, 100)
    print(f"Independent mutual information: {discrete_mutual_information(x, y):.3f}")
    y = np.clip(x + np.random.choice(2, 100), 0, 1)
    print(f"Informative mutual information: {discrete_mutual_information(x, y):.3f}")