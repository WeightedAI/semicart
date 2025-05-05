"""
Evaluation metrics for model assessment.

This module provides functions for evaluating model performance using 
various classification metrics.
"""

from typing import Dict, Optional, Union, Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)


def evaluate_model(
    y_true: NDArray, 
    y_pred: NDArray, 
    y_score: Optional[NDArray] = None, 
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Evaluate a model using multiple classification metrics.
    
    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    y_score : array-like, optional
        Probability estimates, required for AUC calculation.
    average : str, optional (default='macro')
        Averaging strategy for multiclass metrics.
        
    Returns
    -------
    dict
        Dictionary of evaluation metrics including accuracy, precision, 
        recall, F1 score, and possibly AUC (if y_score is provided).
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    # Add AUC if probability estimates are provided
    if y_score is not None:
        try:
            # For binary classification
            if len(np.unique(y_true)) == 2:
                metrics['auc'] = roc_auc_score(y_true, y_score[:, 1])
            # For multiclass
            else:
                metrics['auc'] = roc_auc_score(y_true, y_score, multi_class='ovr', average=average)
        except Exception:
            metrics['auc'] = np.nan
    
    return metrics 