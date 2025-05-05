"""
Command-line interface for SemiCART.

This module provides a CLI for using the SemiCART algorithm directly from the command line.
"""

import argparse
import logging
import sys
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from semicart import SemiCART
from semicart.metrics.evaluation import evaluate_model
from semicart.utils.logging import get_logger, logger


def get_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset by name.
    
    Parameters
    ----------
    name : str
        Name of the dataset to load.
        
    Returns
    -------
    tuple
        (X, y) features and target.
        
    Raises
    ------
    ValueError
        If the dataset name is not recognized.
    """
    # Built-in sklearn datasets
    if name.lower() == 'iris':
        data = load_iris()
        return data.data, data.target
    elif name.lower() == 'wine':
        data = load_wine()
        return data.data, data.target
    elif name.lower() == 'breast_cancer':
        data = load_breast_cancer()
        return data.data, data.target
    else:
        # Try to load from file
        try:
            data = pd.read_csv(name)
            if 'target' in data.columns:
                y = data['target'].values
                X = data.drop('target', axis=1).values
            else:
                # Assume last column is target
                y = data.iloc[:, -1].values
                X = data.iloc[:, :-1].values
            return X, y
        except Exception as e:
            raise ValueError(f"Unknown dataset: {name}. Error: {e}")


def run_experiment(
    dataset_name: str,
    test_size: float,
    k_neighbors: int,
    distance_metric: str,
    max_depth: Optional[int],
    random_state: int,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a SemiCART experiment and compare with standard CART.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to use.
    test_size : float
        Proportion of the dataset to use as test set.
    k_neighbors : int
        Number of neighbors to use for SemiCART.
    distance_metric : str
        Distance metric to use.
    max_depth : int or None
        Maximum depth of the tree.
    random_state : int
        Random state for reproducibility.
    output_file : str, optional
        Path to save results as CSV.
        
    Returns
    -------
    dict
        Dictionary of experiment results.
    """
    logger.info(f"Loading dataset {dataset_name}")
    X, y = get_dataset(dataset_name)
    
    # Preprocessing
    logger.info("Preprocessing data")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    logger.info(f"Splitting data with test_size={test_size}")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, 
        random_state=random_state, 
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Train and evaluate SemiCART
    logger.info(f"Training SemiCART with k_neighbors={k_neighbors}, distance={distance_metric}")
    semicart = SemiCART(
        k_neighbors=k_neighbors, 
        distance_metric=distance_metric,
        max_depth=max_depth,
        random_state=random_state
    )
    semicart.fit(X_train, y_train, X_test)
    y_pred_semi = semicart.predict(X_test)
    try:
        y_proba_semi = semicart.predict_proba(X_test)
    except:
        y_proba_semi = None
    
    # Train and evaluate standard CART for comparison
    from sklearn.tree import DecisionTreeClassifier
    logger.info("Training standard CART for comparison")
    cart = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    cart.fit(X_train, y_train)
    y_pred_cart = cart.predict(X_test)
    try:
        y_proba_cart = cart.predict_proba(X_test)
    except:
        y_proba_cart = None
    
    # Evaluate models
    logger.info("Evaluating models")
    cart_metrics = evaluate_model(y_test, y_pred_cart, y_proba_cart)
    semi_metrics = evaluate_model(y_test, y_pred_semi, y_proba_semi)
    
    # Calculate improvements
    improvements = {
        f'improvement_{key}': semi_metrics[key] - cart_metrics[key]
        for key in cart_metrics
    }
    
    # Organize results
    results = {
        'dataset': dataset_name,
        'test_size': test_size,
        'k_neighbors': k_neighbors,
        'distance_metric': distance_metric,
        'max_depth': max_depth,
    }
    
    # Add metrics for both models
    for key, value in cart_metrics.items():
        results[f'cart_{key}'] = value
    
    for key, value in semi_metrics.items():
        results[f'semi_{key}'] = value
    
    # Add improvements
    results.update(improvements)
    
    # Save results to CSV if output file specified
    if output_file:
        pd.DataFrame([results]).to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    
    # Print results summary
    logger.info("\n----- Results Summary -----")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Test size: {test_size}")
    logger.info(f"SemiCART config: k={k_neighbors}, distance={distance_metric}, max_depth={max_depth}")
    logger.info("\nMetrics:")
    for metric in cart_metrics:
        improvement = improvements[f'improvement_{metric}']
        sign = '+' if improvement > 0 else ''
        logger.info(
            f"  {metric.upper()}: CART={cart_metrics[metric]:.4f}, "
            f"SemiCART={semi_metrics[metric]:.4f}, "
            f"Diff={sign}{improvement:.4f}"
        )
    
    return results


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="SemiCART: Semi-Supervised Classification and Regression Tree algorithm"
    )
    
    parser.add_argument(
        '--dataset', '-d', 
        type=str, 
        default='iris',
        help='Dataset to use (iris, wine, breast_cancer, or path to CSV file)'
    )
    
    parser.add_argument(
        '--test-size', '-t', 
        type=float, 
        default=0.3,
        help='Proportion of the dataset to use as test set'
    )
    
    parser.add_argument(
        '--k-neighbors', '-k', 
        type=int, 
        default=5,
        help='Number of neighbors to use for SemiCART'
    )
    
    parser.add_argument(
        '--distance-metric', '-m', 
        type=str, 
        default='euclidean',
        help='Distance metric to use (euclidean, manhattan, cosine, etc.)'
    )
    
    parser.add_argument(
        '--max-depth', 
        type=int, 
        default=None,
        help='Maximum depth of the tree (None for unlimited)'
    )
    
    parser.add_argument(
        '--random-state', 
        type=int, 
        default=42,
        help='Random state for reproducibility'
    )
    
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        default=None,
        help='Path to save results as CSV'
    )
    
    parser.add_argument(
        '--verbose', '-v', 
        action='count', 
        default=0,
        help='Increase verbosity (can be used multiple times)'
    )
    
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.verbose == 0:
        log_level = logging.INFO
    elif args.verbose == 1:
        log_level = logging.DEBUG
    else:
        log_level = logging.DEBUG
    
    logger.setLevel(log_level)
    
    # Run experiment
    try:
        run_experiment(
            dataset_name=args.dataset,
            test_size=args.test_size,
            k_neighbors=args.k_neighbors,
            distance_metric=args.distance_metric,
            max_depth=args.max_depth,
            random_state=args.random_state,
            output_file=args.output
        )
    except Exception as e:
        logger.error(f"Error during experiment: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 