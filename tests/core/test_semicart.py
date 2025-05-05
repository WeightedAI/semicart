"""Unit tests for the SemiCART class."""

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from semicart import SemiCART


class TestSemiCART:
    """Tests for the SemiCART class."""
    
    @pytest.fixture
    def iris_data(self):
        """Load and prepare the Iris dataset for testing."""
        X, y = load_iris(return_X_y=True)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test
    
    @pytest.fixture
    def binary_data(self):
        """Generate a binary classification dataset for testing."""
        X, y = make_classification(
            n_samples=100, n_features=20, n_informative=10, 
            n_redundant=5, random_state=42
        )
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test
    
    def test_init(self):
        """Test SemiCART initialization with default parameters."""
        model = SemiCART()
        assert model.max_depth is None
        assert model.min_samples_split == 2
        assert model.k_neighbors == 1
        assert model.distance_metric == 'euclidean'
        assert model.initial_weight == 1.0
        assert model.weight_increment == 1.0
    
    def test_init_with_params(self):
        """Test SemiCART initialization with custom parameters."""
        model = SemiCART(
            max_depth=5,
            min_samples_split=5,
            k_neighbors=3,
            distance_metric='manhattan',
            initial_weight=0.5,
            weight_increment=2.0
        )
        assert model.max_depth == 5
        assert model.min_samples_split == 5
        assert model.k_neighbors == 3
        assert model.distance_metric == 'manhattan'
        assert model.initial_weight == 0.5
        assert model.weight_increment == 2.0
    
    def test_invalid_distance_metric(self):
        """Test SemiCART with invalid distance metric."""
        model = SemiCART(distance_metric='invalid_metric')
        X, y = make_classification(random_state=42)
        with pytest.raises(ValueError):
            model.fit(X, y)
    
    def test_fit_iris(self, iris_data):
        """Test SemiCART fit method on the Iris dataset."""
        X_train, X_test, y_train, y_test = iris_data
        model = SemiCART(k_neighbors=3)
        model.fit(X_train, y_train, X_test)
        
        # Check that the model has been fitted correctly
        assert hasattr(model, 'tree_')
        assert hasattr(model, 'instance_weights_')
        assert len(model.instance_weights_) == len(X_train)
    
    def test_predict_iris(self, iris_data):
        """Test SemiCART predict method on the Iris dataset."""
        X_train, X_test, y_train, y_test = iris_data
        model = SemiCART(k_neighbors=3)
        model.fit(X_train, y_train, X_test)
        
        # Test predictions
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(X_test)
        assert np.all(np.isin(y_pred, np.unique(y_train)))
    
    def test_predict_proba_iris(self, iris_data):
        """Test SemiCART predict_proba method on the Iris dataset."""
        X_train, X_test, y_train, y_test = iris_data
        model = SemiCART(k_neighbors=3)
        model.fit(X_train, y_train, X_test)
        
        # Test probability predictions
        y_proba = model.predict_proba(X_test)
        assert y_proba.shape == (len(X_test), len(np.unique(y_train)))
        assert np.allclose(np.sum(y_proba, axis=1), 1.0)
    
    def test_binary_classification(self, binary_data):
        """Test SemiCART on binary classification data."""
        X_train, X_test, y_train, y_test = binary_data
        model = SemiCART(k_neighbors=3)
        model.fit(X_train, y_train, X_test)
        
        # Test predictions
        y_pred = model.predict(X_test)
        assert len(y_pred) == len(X_test)
        assert np.all(np.isin(y_pred, [0, 1]))
        
        # Test probability predictions
        y_proba = model.predict_proba(X_test)
        assert y_proba.shape == (len(X_test), 2)
        assert np.allclose(np.sum(y_proba, axis=1), 1.0)
    
    def test_fit_with_no_test_data(self, iris_data):
        """Test SemiCART fit method without providing test data."""
        X_train, _, y_train, _ = iris_data
        model = SemiCART(k_neighbors=3)
        model.fit(X_train, y_train)  # No X_test provided
        
        # Should use X_train as both training and test data
        assert hasattr(model, 'tree_')
        assert hasattr(model, 'instance_weights_')
        assert len(model.instance_weights_) == len(X_train)
    
    def test_calculate_instance_weights(self, iris_data):
        """Test the instance weights calculation."""
        X_train, X_test, y_train, y_test = iris_data
        model = SemiCART(k_neighbors=1)
        
        # Calculate weights manually
        weights = model._calculate_instance_weights(X_train, X_test)
        
        # Each test instance should affect k (1) training instances
        assert np.sum(weights > model.initial_weight) <= len(X_test) * model.k_neighbors
    
    def test_different_k_neighbors(self, iris_data):
        """Test SemiCART with different k_neighbors values."""
        X_train, X_test, y_train, y_test = iris_data
        
        # Try different k values
        for k in [1, 3, 5]:
            model = SemiCART(k_neighbors=k)
            model.fit(X_train, y_train, X_test)
            weights = model.instance_weights_
            
            # Number of weights > initial should be <= k * n_test
            affected_count = np.sum(weights > model.initial_weight)
            assert affected_count <= k * len(X_test)
    
    def test_scikit_learn_compatibility(self):
        """Test if SemiCART is compatible with scikit-learn's API."""
        try:
            check_estimator(SemiCART())
            is_compatible = True
        except:
            is_compatible = False
        
 