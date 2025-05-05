"""
Distance utility functions for SemiCART.

This module provides utilities for working with various distance metrics
used in the SemiCART algorithm.
"""

from typing import Callable, Dict, Optional, Union, Any, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import (
    euclidean_distances, manhattan_distances, cosine_distances
)
from scipy.spatial.distance import (
    braycurtis, canberra, chebyshev, cityblock, correlation, 
    dice, euclidean, hamming, jaccard, jensenshannon, 
    minkowski, sqeuclidean, yule
)


def pairwise_distance_wrapper(dist_func: Callable[[NDArray, NDArray], float]) -> Callable:
    """
    Wrapper for scipy distance functions to make them work with pairwise distances.
    
    Parameters
    ----------
    dist_func : callable
        Distance function that takes two vectors and returns a scalar.
        
    Returns
    -------
    callable
        Function that takes two matrices and returns a pairwise distance matrix.
    """
    def pairwise_dist(X: NDArray, Y: NDArray) -> NDArray:
        """Compute pairwise distances between each row of X and Y."""
        result = np.zeros((X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                try:
                    result[i, j] = dist_func(X[i], Y[j])
                except Exception:
                    # If calculation fails, use euclidean distance
                    result[i, j] = euclidean(X[i], Y[j])
        return result
    return pairwise_dist


# Dictionary of available distance functions
DISTANCE_FUNCTIONS: Dict[str, Callable] = {
    'euclidean': euclidean_distances,
    'manhattan': manhattan_distances,
    'cosine': cosine_distances,
    'braycurtis': pairwise_distance_wrapper(braycurtis),
    'canberra': pairwise_distance_wrapper(canberra),
    'chebyshev': pairwise_distance_wrapper(chebyshev),
    'cityblock': pairwise_distance_wrapper(cityblock),
    'correlation': pairwise_distance_wrapper(correlation),
    'dice': pairwise_distance_wrapper(dice),
    'hamming': pairwise_distance_wrapper(hamming),
    'jaccard': pairwise_distance_wrapper(jaccard),
    'jensenshannon': pairwise_distance_wrapper(jensenshannon),
    'minkowski': lambda X, Y: pairwise_distance_wrapper(
        lambda x, y: minkowski(x, y, 3))(X, Y),
    'sqeuclidean': pairwise_distance_wrapper(sqeuclidean),
    'yule': pairwise_distance_wrapper(yule)
}


def get_distance_function(distance_metric: str) -> Callable:
    """
    Get the distance function for the specified metric.
    
    Parameters
    ----------
    distance_metric : str
        Name of the distance metric.
        Supported values: 'euclidean', 'manhattan', 'cosine', 'braycurtis', 
        'canberra', 'chebyshev', 'cityblock', 'correlation', 'dice', 'hamming',
        'jaccard', 'jensenshannon', 'minkowski', 'sqeuclidean', 'yule'
        
    Returns
    -------
    callable
        The distance function that computes pairwise distances between samples.
        
    Raises
    ------
    ValueError
        If the distance metric is not supported.
    """
    if distance_metric not in DISTANCE_FUNCTIONS:
        valid_metrics = sorted(DISTANCE_FUNCTIONS.keys())
        raise ValueError(
            f"Unknown distance metric: {distance_metric}. "
            f"Valid options are: {', '.join(valid_metrics)}"
        )
    
    return DISTANCE_FUNCTIONS[distance_metric] 