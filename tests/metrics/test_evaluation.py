"""Unit tests for the evaluation metrics module."""

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)

from semicart.metrics.evaluation import evaluate_model


class TestEvaluationMetrics:
    """Tests for the evaluation metrics functions."""
    
    @pytest.fixture
    def binary_data(self):
        """Create binary classification test data."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        y_score = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.6, 0.4],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.7, 0.3],
            [0.2, 0.8]
        ])
        return y_true, y_pred, y_score
    
    @pytest.fixture
    def multiclass_data(self):
        """Create multiclass classification test data."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 0, 2, 0, 1, 2])
        y_score = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.6, 0.3],
            [0.7, 0.2, 0.1],
            [0.3, 0.4, 0.3],
            [0.1, 0.1, 0.8],
            [0.9, 0.0, 0.1],
            [0.2, 0.7, 0.1],
            [0.0, 0.2, 0.8]
        ])
        return y_true, y_pred, y_score
    
    def test_evaluate_model_binary(self, binary_data):
        """Test evaluate_model function with binary classification data."""
        y_true, y_pred, y_score = binary_data
        
        # Evaluate with our function
        metrics = evaluate_model(y_true, y_pred, y_score)
        
        # Calculate expected metrics directly
        expected_accuracy = accuracy_score(y_true, y_pred)
        expected_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        expected_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        expected_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        expected_auc = roc_auc_score(y_true, y_score[:, 1])
        
        # Check that our metrics match the expected values
        assert metrics['accuracy'] == pytest.approx(expected_accuracy)
        assert metrics['precision'] == pytest.approx(expected_precision)
        assert metrics['recall'] == pytest.approx(expected_recall)
        assert metrics['f1'] == pytest.approx(expected_f1)
        assert metrics['auc'] == pytest.approx(expected_auc)
    
    def test_evaluate_model_multiclass(self, multiclass_data):
        """Test evaluate_model function with multiclass classification data."""
        y_true, y_pred, y_score = multiclass_data
        
        # Evaluate with our function
        metrics = evaluate_model(y_true, y_pred, y_score)
        
        # Calculate expected metrics directly
        expected_accuracy = accuracy_score(y_true, y_pred)
        expected_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        expected_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        expected_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        expected_auc = roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
        
        # Check that our metrics match the expected values
        assert metrics['accuracy'] == pytest.approx(expected_accuracy)
        assert metrics['precision'] == pytest.approx(expected_precision)
        assert metrics['recall'] == pytest.approx(expected_recall)
        assert metrics['f1'] == pytest.approx(expected_f1)
        assert metrics['auc'] == pytest.approx(expected_auc)
    
    def test_evaluate_model_without_scores(self, binary_data):
        """Test evaluate_model function without probability scores."""
        y_true, y_pred, _ = binary_data
        
        # Evaluate without scores
        metrics = evaluate_model(y_true, y_pred)
        
        # Check that metrics are calculated correctly
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' not in metrics
    
    def test_evaluate_model_with_different_average(self, multiclass_data):
        """Test evaluate_model function with different averaging methods."""
        y_true, y_pred, y_score = multiclass_data
        
        # Test with different average parameters
        for average in ['micro', 'macro', 'weighted']:
            metrics = evaluate_model(y_true, y_pred, y_score, average=average)
            
            # Calculate expected values
            expected_precision = precision_score(y_true, y_pred, average=average, zero_division=0)
            expected_recall = recall_score(y_true, y_pred, average=average, zero_division=0)
            expected_f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
            
            # Check that our metrics match the expected values
            assert metrics['precision'] == pytest.approx(expected_precision)
            assert metrics['recall'] == pytest.approx(expected_recall)
            assert metrics['f1'] == pytest.approx(expected_f1)
    
    def test_evaluate_model_with_invalid_scores(self, binary_data):
        """Test evaluate_model with invalid probability scores."""
        y_true, y_pred, _ = binary_data
        
        # Create invalid scores (wrong shape)
        invalid_y_score = np.random.rand(len(y_true))
        
        # Should not raise an exception, just return NaN for AUC
        metrics = evaluate_model(y_true, y_pred, invalid_y_score)
        
        assert 'auc' in metrics
        assert np.isnan(metrics['auc'])
    
    def test_evaluate_model_all_incorrect(self):
        """Test evaluate_model with all incorrect predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        
        metrics = evaluate_model(y_true, y_pred)
        
        # Check metrics for all-incorrect case
        assert metrics['accuracy'] == 0.0
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1'] == 0.0
    
    def test_evaluate_model_all_correct(self):
        """Test evaluate_model with all correct predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        
        metrics = evaluate_model(y_true, y_pred)
        
        # Check metrics for all-correct case
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0 