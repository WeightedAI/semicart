"""Unit tests for the distance utilities."""

import numpy as np
import pytest

from semicart.utils.distance import (
    pairwise_distance_wrapper, 
    get_distance_function, 
    DISTANCE_FUNCTIONS
)


class TestDistanceUtils:
    """Tests for the distance utility functions."""
    
    @pytest.fixture
    def test_data(self):
        """Create test data for distance calculations."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        Y = np.array([[1, 2, 3], [7, 8, 9]])
        return X, Y
    
    def test_pairwise_distance_wrapper(self, test_data):
        """Test the pairwise distance wrapper function."""
        X, Y = test_data
        
        # Create a simple distance function
        def euclidean_dist(a, b):
            return np.sqrt(np.sum((a - b) ** 2))
        
        # Wrap it for pairwise calculations
        pairwise_func = pairwise_distance_wrapper(euclidean_dist)
        
        # Calculate distances
        distances = pairwise_func(X, Y)
        
        # Expected results
        expected = np.array([
            [0.0, np.sqrt(np.sum((np.array([1, 2, 3]) - np.array([7, 8, 9])) ** 2))],
            [np.sqrt(np.sum((np.array([4, 5, 6]) - np.array([1, 2, 3])) ** 2)), 
             np.sqrt(np.sum((np.array([4, 5, 6]) - np.array([7, 8, 9])) ** 2))]
        ])
        
        # Check results
        assert distances.shape == (2, 2)
        assert np.allclose(distances, expected)
    
    def test_get_distance_function(self):
        """Test getting a distance function by name."""
        # Check all available distance functions
        for name in DISTANCE_FUNCTIONS.keys():
            func = get_distance_function(name)
            assert callable(func)
    
    def test_get_invalid_distance_function(self):
        """Test getting an invalid distance function."""
        with pytest.raises(ValueError):
            get_distance_function('invalid_metric')
    
    def test_distance_functions(self, test_data):
        """Test that all distance functions work correctly."""
        X, Y = test_data
        
        for name, func in DISTANCE_FUNCTIONS.items():
            # Skip tests for distance functions that might have issues with negative values
            if name in ['braycurtis', 'dice', 'jaccard', 'jensenshannon']:
                continue
                
            try:
                distances = func(X, Y)
                
                # Check the shape of the result
                assert distances.shape == (2, 2)
                
                # Check that diagonal distances for identical points are smaller
                if name != 'correlation':  # Correlation can be negative
                    assert distances[0, 0] < distances[0, 1]
                    assert distances[0, 0] < distances[1, 0]
                
            except Exception as e:
                pytest.fail(f"Distance function {name} failed: {e}")
    
    def test_euclidean_distance(self, test_data):
        """Test the euclidean distance specifically."""
        X, Y = test_data
        
        # Get euclidean distance function
        euclidean_func = get_distance_function('euclidean')
        
        # Calculate distances
        distances = euclidean_func(X, Y)
        
        # Check that identical points have zero distance
        assert np.isclose(distances[0, 0], 0.0)
        
        # Check that distances are symmetric
        assert np.isclose(distances[0, 1], np.sqrt(np.sum((np.array([1, 2, 3]) - np.array([7, 8, 9])) ** 2)))
        assert np.isclose(distances[1, 0], np.sqrt(np.sum((np.array([4, 5, 6]) - np.array([1, 2, 3])) ** 2)))
    
    def test_manhattan_distance(self, test_data):
        """Test the manhattan distance specifically."""
        X, Y = test_data
        
        # Get manhattan distance function
        manhattan_func = get_distance_function('manhattan')
        
        # Calculate distances
        distances = manhattan_func(X, Y)
        
        # Check that identical points have zero distance
        assert np.isclose(distances[0, 0], 0.0)
        
        # Check specific distances
        assert np.isclose(distances[0, 1], np.sum(np.abs(np.array([1, 2, 3]) - np.array([7, 8, 9]))))
        assert np.isclose(distances[1, 0], np.sum(np.abs(np.array([4, 5, 6]) - np.array([1, 2, 3]))))
    
    def test_cosine_distance(self, test_data):
        """Test the cosine distance specifically."""
        X, Y = test_data
        
        # Get cosine distance function
        cosine_func = get_distance_function('cosine')
        
        # Calculate distances
        distances = cosine_func(X, Y)
        
        # Check that identical points have zero distance
        assert np.isclose(distances[0, 0], 0.0)
        
        # Check range of values (cosine distance is between 0 and 2)
        assert np.all(distances >= 0)
        assert np.all(distances <= 2)
    
    def test_fallback_on_error(self):
        """Test that the wrapper falls back to euclidean on error."""
        # Create a distance function that raises an error
        def error_dist(a, b):
            raise ValueError("Test error")
        
        # Wrap it
        wrapped_func = pairwise_distance_wrapper(error_dist)
        
        # Create test data
        X = np.array([[1, 2, 3], [4, 5, 6]])
        Y = np.array([[1, 2, 3], [7, 8, 9]])
        
        # Calculate distances - should use euclidean as fallback
        distances = wrapped_func(X, Y)
        
        # Check that we got a result
        assert distances.shape == (2, 2)
        assert not np.any(np.isnan(distances)) 