"""
Semi-Supervised Classification and Regression Tree (SemiCART) implementation.

This module contains the implementation of the SemiCART algorithm, which enhances
traditional CART by incorporating a distance-based weighting method that assigns
weights to training instances based on their proximity to test instances.
"""

from typing import Callable, Dict, Optional, List, Union, Any
import logging

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.tree import DecisionTreeClassifier

from semicart.utils.distance import get_distance_function, DISTANCE_FUNCTIONS
from semicart.utils.logging import logger


class SemiCART(BaseEstimator, ClassifierMixin):
    """
    Semi-CART: Semi-Supervised Classification and Regression Tree algorithm
    
    This algorithm enhances traditional CART by incorporating a distance-based weighting 
    method that assigns weights to training instances based on their proximity to test instances.
    
    Parameters
    ----------
    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, nodes are expanded until all leaves
        are pure or until all leaves contain less than min_samples_split samples.
        
    min_samples_split : int or float, optional (default=2)
        The minimum number of samples required to split an internal node.
        
    k_neighbors : int, optional (default=1)
        Number of nearest neighbors to consider for weight assignment.
        
    distance_metric : str, optional (default='euclidean')
        The distance metric to use for weight calculation.
        Supported values: 'euclidean', 'manhattan', 'cosine', 'braycurtis', 
        'canberra', 'chebyshev', 'cityblock', 'correlation', 'dice', 'hamming',
        'jaccard', 'jensenshannon', 'minkowski', 'sqeuclidean', 'yule'
        
    initial_weight : float, optional (default=1.0)
        The initial weight assigned to each training instance.
        
    weight_increment : float, optional (default=1.0)
        The increment added to a training instance's weight when it's found to be a nearest neighbor.
        
    random_state : int, optional (default=None)
        Random state for reproducibility.
        
    log_level : int, optional (default=logging.INFO)
        Logging level for the SemiCART instance.
        
    Attributes
    ----------
    tree_ : DecisionTreeClassifier
        The underlying decision tree classifier.
        
    instance_weights_ : ndarray of shape (n_samples,)
        The calculated weights for each training instance.
        
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
        
    Examples
    --------
    >>> from semicart import SemiCART
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    >>> model = SemiCART(k_neighbors=3, distance_metric='euclidean')
    >>> model.fit(X_train, y_train, X_test)
    >>> y_pred = model.predict(X_test)
    """
    
    def __init__(
        self, 
        max_depth: Optional[int] = None, 
        min_samples_split: Union[int, float] = 2, 
        k_neighbors: int = 1, 
        distance_metric: str = 'euclidean', 
        initial_weight: float = 1.0, 
        weight_increment: float = 1.0,
        random_state: Optional[int] = None,
        log_level: int = logging.INFO
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.k_neighbors = k_neighbors
        self.distance_metric = distance_metric
        self.initial_weight = initial_weight
        self.weight_increment = weight_increment
        self.random_state = random_state
        self.log_level = log_level
        
        # Initialize instance-specific logger
        self.logger = logger.getChild(self.__class__.__name__)
        self.logger.setLevel(self.log_level)
    
    def _calculate_instance_weights(
        self, 
        X_train: NDArray, 
        X_test: NDArray
    ) -> NDArray:
        """
        Calculate weights for training instances based on their proximity to test instances.
        
        Parameters
        ----------
        X_train : ndarray of shape (n_train_samples, n_features)
            The training input samples.
            
        X_test : ndarray of shape (n_test_samples, n_features)
            The test input samples.
            
        Returns
        -------
        weights : ndarray of shape (n_train_samples,)
            The calculated weights for each training instance.
            
        Raises
        ------
        ValueError
            If the specified distance metric is not supported.
        """
        # Initialize weights
        n_train_samples = X_train.shape[0]
        weights = np.ones(n_train_samples) * self.initial_weight
        
        # Get the distance function for the specified metric
        try:
            distance_func = get_distance_function(self.distance_metric)
            self.logger.debug(f"Using distance metric: {self.distance_metric}")
            
            # Calculate pairwise distances between training and test instances
            distances = distance_func(X_train, X_test)
        except Exception as e:
            self.logger.error(f"Error with {self.distance_metric} distance: {e}")
            raise ValueError(
                f"Unknown distance metric: {self.distance_metric}. "
                f"Valid options are: {', '.join(sorted(DISTANCE_FUNCTIONS.keys()))}"
            )
        
        # For each test instance, find the k nearest training instances
        for j in range(X_test.shape[0]):
            # Get indices of k nearest neighbors in training set
            nearest_indices = np.argsort(distances[:, j])[:self.k_neighbors]
            
            # Increment weights of these neighbors
            weights[nearest_indices] += self.weight_increment
        
        self.logger.debug(
            f"Instance weights calculated. Min: {weights.min():.2f}, "
            f"Max: {weights.max():.2f}, Mean: {weights.mean():.2f}"
        )
        
        return weights
    
    def _modified_gini(
        self, 
        y: NDArray, 
        sample_weights: NDArray
    ) -> float:
        """
        Calculate the modified Gini index incorporating instance weights.
        
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The class labels.
            
        sample_weights : ndarray of shape (n_samples,)
            The weight of each sample.
            
        Returns
        -------
        gini : float
            The modified Gini index.
        """
        if len(y) == 0:
            return 0.0
            
        # Get unique classes and their weighted counts
        classes, y_counts = np.unique(y, return_counts=True)
        total_weight = np.sum(sample_weights)
        
        # If all weights are zero, return 0
        if total_weight == 0:
            return 0.0
        
        # Calculate weighted class proportions
        weighted_proportions = np.zeros(len(classes))
        for i, cls in enumerate(classes):
            cls_indices = (y == cls)
            cls_weight = np.sum(sample_weights[cls_indices])
            weighted_proportions[i] = cls_weight / total_weight
        
        # Calculate Gini index
        gini = 1.0 - np.sum(weighted_proportions ** 2)
        
        return gini
    
    def fit(
        self, 
        X: NDArray, 
        y: NDArray, 
        X_test: Optional[NDArray] = None
    ) -> "SemiCART":
        """
        Fit the Semi-CART model according to the given training data.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
            
        y : array-like of shape (n_samples,)
            The target values.
            
        X_test : {array-like, sparse matrix} of shape (n_test_samples, n_features), optional
            The test input samples. If not provided, X will be used as both training and test data.
            
        Returns
        -------
        self : object
            Returns self.
            
        Raises
        ------
        ValueError
            If inputs are invalid.
        """
        # Input validation
        X, y = check_X_y(X, y)
        
        # Store classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        self.logger.info(
            f"Fitting SemiCART with k_neighbors={self.k_neighbors}, "
            f"distance_metric={self.distance_metric}, n_classes={self.n_classes_}"
        )
        
        # If no test data is provided, use the training data
        if X_test is None:
            self.logger.info("No test data provided. Using training data as test data.")
            X_test = X
        else:
            X_test = check_array(X_test)
            self.logger.info(f"Using provided test data of shape {X_test.shape}")
        
        # Calculate instance weights based on proximity to test data
        self.instance_weights_ = self._calculate_instance_weights(X, X_test)
        
        # Filter out instances with zero weight
        non_zero_weight_indices = np.where(self.instance_weights_ > 0)[0]
        
        # If all instances have zero weight (unlikely but possible), keep all instances
        if len(non_zero_weight_indices) == 0:
            self.logger.warning("All instances have zero weight. Using uniform weights.")
            X_filtered = X
            y_filtered = y
            weights_filtered = np.ones(X.shape[0])
        else:
            X_filtered = X[non_zero_weight_indices]
            y_filtered = y[non_zero_weight_indices]
            weights_filtered = self.instance_weights_[non_zero_weight_indices]
            self.logger.info(
                f"Using {len(X_filtered)} instances with non-zero weights "
                f"out of {len(X)} total instances"
            )
        
        # Create and fit a decision tree with the modified sample weights
        self.tree_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state
        )
        
        self.tree_.fit(X_filtered, y_filtered, sample_weight=weights_filtered)
        self.logger.info(f"Decision tree fitted with max_depth={self.max_depth}")
        
        # Store training data info
        self.n_features_in_ = X.shape[1]
        
        return self
    
    def predict(self, X: NDArray) -> NDArray:
        """
        Predict class for X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
            
        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        """
        # Check if fit has been called
        check_is_fitted(self, ['tree_', 'instance_weights_'])
        
        # Input validation
        X = check_array(X)
        
        # Check feature dimension
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SemiCART is expecting "
                f"{self.n_features_in_} features"
            )
        
        self.logger.debug(f"Predicting classes for {X.shape[0]} samples")
        
        # Predict using the underlying tree
        return self.tree_.predict(X)
    
    def predict_proba(self, X: NDArray) -> NDArray:
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
            
        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        """
        # Check if fit has been called
        check_is_fitted(self, ['tree_', 'instance_weights_'])
        
        # Input validation
        X = check_array(X)
        
        # Check feature dimension
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SemiCART is expecting "
                f"{self.n_features_in_} features"
            )
        
        self.logger.debug(f"Predicting probabilities for {X.shape[0]} samples")
        
        # Predict probabilities using the underlying tree
        return self.tree_.predict_proba(X)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return super().get_params(deep=deep)
    
    def set_params(self, **params: Any) -> "SemiCART":
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
            
        Returns
        -------
        self : object
            Estimator instance.
        """
        return super().set_params(**params) 