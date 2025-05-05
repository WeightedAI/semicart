"""
Benchmarking utilities for SemiCART.

This module provides functionality for benchmarking SemiCART against
standard CART on various datasets with different configurations.
"""

import os
import time
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from semicart import SemiCART
from semicart.metrics.evaluation import evaluate_model
from semicart.utils.logging import logger
from semicart.utils.distance import DISTANCE_FUNCTIONS


class BenchmarkRunner:
    """
    Runner for SemiCART benchmarking experiments.
    
    This class provides functionality to run comprehensive benchmarks
    comparing SemiCART with standard CART across various datasets,
    test sizes, numbers of neighbors, and distance metrics.
    
    Parameters
    ----------
    output_dir : str, optional (default='results')
        Directory to save benchmark results.
    random_state : int, optional (default=42)
        Random state for reproducibility.
    log_level : int, optional (default=logging.INFO)
        Logging level.
    """
    
    def __init__(
        self, 
        output_dir: str = 'results',
        random_state: int = 42,
        log_level: int = logging.INFO
    ):
        self.output_dir = output_dir
        self.random_state = random_state
        self.log_level = log_level
        
        # Initialize logger
        self.logger = logger.getChild(self.__class__.__name__)
        self.logger.setLevel(log_level)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = []
        
        # Suppress warnings during benchmarking
        warnings.filterwarnings('ignore')
    
    def get_dataset(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
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
        """
        # Built-in sklearn datasets
        if name == 'iris':
            data = load_iris()
            return data.data, data.target
        elif name == 'wine':
            data = load_wine()
            return data.data, data.target
        elif name == 'breast_cancer':
            data = load_breast_cancer()
            return data.data, data.target
        
        # UCI datasets through OpenML
        try:
            if name == 'banknote':
                data = fetch_openml(name='banknote-authentication', version=1, as_frame=False)
            elif name == 'fertility':
                data = fetch_openml(name='fertility', version=1, as_frame=False)
            elif name == 'wdbc':
                data = fetch_openml(name='wdbc', version=1, as_frame=False)
            elif name == 'biodeg':
                data = fetch_openml(name='biodeg', version=1, as_frame=False)
            elif name == 'haberman':
                data = fetch_openml(name='haberman', version=1, as_frame=False)
            elif name == 'transfusion':
                data = fetch_openml(name='blood-transfusion-service-center', version=1, as_frame=False)
            elif name == 'hepatitis':
                data = fetch_openml(name='hepatitis', version=1, as_frame=False)
            elif name == 'tictactoe':
                data = fetch_openml(name='tic-tac-toe', version=1, as_frame=False)
            elif name == 'vote':
                data = fetch_openml(name='vote', version=1, as_frame=False)
            elif name == 'bupa':
                data = fetch_openml(name='liver-disorders', version=1, as_frame=False)
            elif name == 'breast':
                data = fetch_openml(name='breast-cancer-wisconsin', version=1, as_frame=False)
            elif name == 'glass':
                data = fetch_openml(name='glass', version=1, as_frame=False)
            elif name == 'mammographic_masses':
                data = fetch_openml(name='mammographic', version=1, as_frame=False)
            else:
                raise ValueError(f"Unknown dataset: {name}")
            
            # Preprocess the dataset
            X = data.data
            
            # Convert target to numeric if needed
            if hasattr(data, 'target') and data.target is not None:
                y = data.target
                if isinstance(y[0], str):
                    # Encode string targets to integers
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                else:
                    # Convert to int if numeric
                    y = y.astype(int)
            else:
                raise ValueError(f"No target found for dataset: {name}")
            
            return X, y
        
        except Exception as e:
            self.logger.error(f"Error loading dataset {name}: {e}")
            raise
    
    def run_comparison(
        self,
        dataset_names: List[str],
        test_sizes: List[float],
        k_neighbors_values: List[int],
        distance_metrics: List[str],
        max_depth: Optional[int] = None,
        min_samples_split: int = 2
    ) -> pd.DataFrame:
        """
        Run a comprehensive comparison of CART vs Semi-CART.
        
        Parameters
        ----------
        dataset_names : list
            List of dataset names.
        test_sizes : list
            List of test sizes to try.
        k_neighbors_values : list
            List of k_neighbors values to try.
        distance_metrics : list
            List of distance metrics to try.
        max_depth : int or None, optional (default=None)
            Maximum depth of the tree.
        min_samples_split : int, optional (default=2)
            Minimum samples required to split a node.
            
        Returns
        -------
        DataFrame
            Results dataframe.
        """
        results = []
        
        self.logger.info(f"Starting comprehensive comparison with:")
        self.logger.info(f"  - {len(dataset_names)} datasets: {', '.join(dataset_names)}")
        self.logger.info(f"  - {len(test_sizes)} test sizes: {', '.join([str(ts) for ts in test_sizes])}")
        self.logger.info(f"  - {len(k_neighbors_values)} k values: {', '.join([str(k) for k in k_neighbors_values])}")
        self.logger.info(f"  - {len(distance_metrics)} distance metrics: {', '.join(distance_metrics)}")
        self.logger.info("\nThis will result in", 
              len(dataset_names) * len(test_sizes) * len(k_neighbors_values) * len(distance_metrics),
              "total model evaluations.")
        
        for dataset_name in dataset_names:
            self.logger.info(f"\nProcessing dataset: {dataset_name}")
            
            try:
                # Load dataset
                X, y = self.get_dataset(dataset_name)
                
                # Handle empty or invalid datasets
                if X is None or y is None or len(X) == 0 or len(y) == 0:
                    self.logger.warning(f"Empty or invalid dataset for {dataset_name}, skipping.")
                    continue
                    
                # Check if X contains NaN or infinite values
                if np.isnan(X).any() or np.isinf(X).any():
                    self.logger.warning(f"Dataset {dataset_name} contains NaN or infinite values. Applying preprocessing.")
                    # Replace NaN with 0 and infinite with large values
                    X = np.nan_to_num(X)
                
                # Scale the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                for test_size in test_sizes:
                    self.logger.info(f"  Test size: {test_size:.1f}")
                    
                    try:
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y, test_size=test_size, 
                            random_state=self.random_state, 
                            stratify=y if len(np.unique(y)) > 1 else None
                        )
                        
                        # Train and evaluate standard CART
                        start_time = time.time()
                        cart = DecisionTreeClassifier(
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            random_state=self.random_state
                        )
                        cart.fit(X_train, y_train)
                        cart_time = time.time() - start_time
                        
                        y_pred_cart = cart.predict(X_test)
                        try:
                            y_proba_cart = cart.predict_proba(X_test)
                        except Exception:
                            y_proba_cart = None
                        
                        cart_metrics = evaluate_model(y_test, y_pred_cart, y_proba_cart)
                        
                        # Try different k_neighbors values and distance metrics
                        for k in k_neighbors_values:
                            for dist_metric in distance_metrics:
                                try:
                                    self.logger.debug(f"    k={k}, distance={dist_metric}")
                                    
                                    # Train and evaluate Semi-CART
                                    start_time = time.time()
                                    semicart = SemiCART(
                                        k_neighbors=k, 
                                        distance_metric=dist_metric,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        random_state=self.random_state,
                                        log_level=logging.WARNING  # Reduce logging during benchmark
                                    )
                                    semicart.fit(X_train, y_train, X_test)
                                    semicart_time = time.time() - start_time
                                    
                                    y_pred_semi = semicart.predict(X_test)
                                    try:
                                        y_proba_semi = semicart.predict_proba(X_test)
                                    except Exception:
                                        y_proba_semi = None
                                    
                                    semi_metrics = evaluate_model(y_test, y_pred_semi, y_proba_semi)
                                    
                                    # Calculate improvements
                                    improvements = {
                                        f'improvement_{key}': semi_metrics[key] - cart_metrics[key]
                                        for key in cart_metrics
                                    }
                                    
                                    # Store results
                                    result = {
                                        'dataset': dataset_name,
                                        'test_size': test_size,
                                        'k_neighbors': k,
                                        'distance_metric': dist_metric,
                                        'cart_time': cart_time,
                                        'semi_time': semicart_time,
                                    }
                                    
                                    # Add metrics for both models
                                    for key, value in cart_metrics.items():
                                        result[f'cart_{key}'] = value
                                    
                                    for key, value in semi_metrics.items():
                                        result[f'semi_{key}'] = value
                                    
                                    # Add improvements
                                    result.update(improvements)
                                    
                                    results.append(result)
                                    
                                    # Print brief result for the best improvement metric (accuracy)
                                    if 'improvement_accuracy' in improvements:
                                        acc_improvement = improvements['improvement_accuracy']
                                        if acc_improvement > 0:
                                            self.logger.info(
                                                f"    k={k}, distance={dist_metric}, "
                                                f"accuracy improvement: +{acc_improvement:.4f}"
                                            )
                                    
                                except Exception as e:
                                    self.logger.error(f"    Error with k={k}, distance={dist_metric}: {e}")
                                    continue
                    
                    except Exception as e:
                        self.logger.error(f"  Error processing test_size={test_size} for dataset {dataset_name}: {e}")
                        continue
            
            except Exception as e:
                self.logger.error(f"Error processing dataset {dataset_name}: {e}")
                continue
        
        # Convert to DataFrame
        if not results:
            self.logger.warning("No results were collected. Check error messages above.")
            return pd.DataFrame()
            
        results_df = pd.DataFrame(results)
        
        # Save results to CSV
        results_path = os.path.join(self.output_dir, 'comparison_results.csv')
        results_df.to_csv(results_path, index=False)
        self.logger.info(f"Results saved to {results_path}")
        
        # Store results for later analysis
        self.results = results_df
        
        self.logger.info("\nComprehensive comparison completed!")
        return results_df
    
    def find_best_configurations(self) -> pd.DataFrame:
        """
        Find the best configurations across datasets and test sizes.
        
        Returns
        -------
        DataFrame
            Best configurations dataframe.
            
        Raises
        ------
        ValueError
            If no results are available to analyze.
        """
        if len(self.results) == 0:
            raise ValueError("No results available. Run comparison first.")
            
        best_configs = []
        
        for dataset in self.results['dataset'].unique():
            for test_size in self.results['test_size'].unique():
                dataset_size_results = self.results[(self.results['dataset'] == dataset) & 
                                               (self.results['test_size'] == test_size)]
                
                # Find best configuration for each metric
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                    if f'semi_{metric}' not in dataset_size_results.columns:
                        continue
                        
                    try:
                        # Find configuration with highest semi metric
                        best_idx = dataset_size_results[f'semi_{metric}'].idxmax()
                        best_row = dataset_size_results.loc[best_idx].copy()
                        
                        # Add best metric information
                        best_row['metric'] = metric
                        best_row['cart_value'] = best_row[f'cart_{metric}']
                        best_row['semi_value'] = best_row[f'semi_{metric}']
                        best_row['improvement'] = best_row[f'improvement_{metric}']
                        
                        best_configs.append(best_row)
                    except Exception:
                        continue
        
        best_configs_df = pd.DataFrame(best_configs)
        
        if best_configs_df.empty:
            self.logger.warning("No best configurations found.")
            return pd.DataFrame()
            
        best_configs_df = best_configs_df[[
            'dataset', 'test_size', 'metric', 'k_neighbors', 'distance_metric',
            'cart_value', 'semi_value', 'improvement'
        ]]
        
        # Save best configurations to CSV
        best_configs_path = os.path.join(self.output_dir, 'best_configurations.csv')
        best_configs_df.to_csv(best_configs_path, index=False)
        self.logger.info(f"Best configurations saved to {best_configs_path}")
        
        return best_configs_df
    
    def plot_best_improvement_heatmap(self, best_configs: pd.DataFrame) -> None:
        """
        Plot a heatmap of the best improvements for each dataset and test size.
        
        Parameters
        ----------
        best_configs : DataFrame
            Best configurations dataframe.
            
        Raises
        ------
        ValueError
            If the dataframe is empty.
        """
        if best_configs.empty:
            raise ValueError("Best configurations dataframe is empty.")
            
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        for metric in metrics:
            # Filter for the specific metric
            metric_data = best_configs[best_configs['metric'] == metric]
            
            if metric_data.empty:
                self.logger.warning(f"No data for metric {metric}")
                continue
                
            # Create pivot table for heatmap
            pivot_data = metric_data.pivot(index='dataset', columns='test_size', values='improvement')
            
            # Plot heatmap
            plt.figure(figsize=(14, 8))
            ax = sns.heatmap(pivot_data, annot=True, cmap='RdBu_r', center=0, fmt='.4f')
            plt.title(f'Best Semi-CART Improvement for {metric.capitalize()}', fontsize=16)
            plt.xlabel('Test Size', fontsize=14)
            plt.ylabel('Dataset', fontsize=14)
            plt.tight_layout()
            
            # Save plot
            fig_path = os.path.join(self.output_dir, f'best_improvement_{metric}_heatmap.png')
            plt.savefig(fig_path)
            plt.close()
            self.logger.info(f"Heatmap for {metric} saved to {fig_path}")
    
    def plot_metric_vs_test_size(
        self, 
        metric: str, 
        dataset_name: str, 
        k_neighbors: Optional[int] = None, 
        distance_metric: Optional[str] = None
    ) -> None:
        """
        Plot a specific metric vs test size.
        
        Parameters
        ----------
        metric : str
            Metric to plot.
        dataset_name : str
            Name of the dataset.
        k_neighbors : int or None, optional
            Number of neighbors (if None, will plot for all).
        distance_metric : str or None, optional
            Distance metric (if None, will use euclidean).
            
        Raises
        ------
        ValueError
            If no results are available for the specified parameters.
        """
        if len(self.results) == 0:
            raise ValueError("No results available. Run comparison first.")
            
        plt.figure(figsize=(12, 8))
        
        # Filter results
        if k_neighbors is not None:
            filtered_results = self.results[(self.results['dataset'] == dataset_name) & 
                                          (self.results['k_neighbors'] == k_neighbors)]
            if distance_metric is not None:
                filtered_results = filtered_results[filtered_results['distance_metric'] == distance_metric]
        else:
            filtered_results = self.results[self.results['dataset'] == dataset_name]
        
        if filtered_results.empty:
            raise ValueError(f"No results for dataset {dataset_name} with the specified parameters.")
            
        # Prepare data for plotting
        test_sizes = sorted(filtered_results['test_size'].unique())
        
        if k_neighbors is None:
            # Plot for all k values with euclidean distance
            k_values = sorted(filtered_results['k_neighbors'].unique())
            for k in k_values:
                k_data = filtered_results[(filtered_results['k_neighbors'] == k) & 
                                         (filtered_results['distance_metric'] == 'euclidean')]
                
                if k_data.empty:
                    continue
                    
                semi_values = [k_data[k_data['test_size'] == ts][f'semi_{metric}'].values[0] 
                              for ts in test_sizes if ts in k_data['test_size'].values]
                plt.plot(test_sizes, semi_values, marker='o', label=f'Semi-CART k={k}')
        else:
            # Plot for specific k with different distance metrics or just euclidean
            if distance_metric is None:
                distance_metrics = sorted(filtered_results['distance_metric'].unique())
                for dm in distance_metrics:
                    dm_data = filtered_results[filtered_results['distance_metric'] == dm]
                    
                    if dm_data.empty:
                        continue
                        
                    semi_values = [dm_data[dm_data['test_size'] == ts][f'semi_{metric}'].values[0] 
                                  for ts in test_sizes if ts in dm_data['test_size'].values]
                    plt.plot(test_sizes, semi_values, marker='o', label=f'Semi-CART {dm}')
            else:
                # Plot for specific k and distance metric
                semi_values = [filtered_results[filtered_results['test_size'] == ts][f'semi_{metric}'].values[0] 
                              for ts in test_sizes if ts in filtered_results['test_size'].values]
                plt.plot(test_sizes, semi_values, marker='o', 
                        label=f'Semi-CART k={k_neighbors}, {distance_metric}')
        
        # Always plot standard CART for comparison
        standard_values = [filtered_results[filtered_results['test_size'] == ts][f'cart_{metric}'].values[0] 
                          for ts in test_sizes if ts in filtered_results['test_size'].values]
        plt.plot(test_sizes, standard_values, marker='s', linestyle='--', color='red', linewidth=2, 
                 label='Standard CART')
        
        plt.title(f'{metric.capitalize()} vs Test Size - {dataset_name} Dataset', fontsize=16)
        plt.xlabel('Test Size', fontsize=14)
        plt.ylabel(metric.capitalize(), fontsize=14)
        plt.xticks(test_sizes)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Add text showing improvement at each test size
        for i, ts in enumerate(test_sizes):
            improvement = semi_values[i] - standard_values[i]
            plt.text(ts, max(semi_values[i], standard_values[i]) + 0.01, 
                    f'{improvement:.4f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        
        # Create filename
        filename = f'{dataset_name.lower()}_{metric}_vs_testsize'
        if k_neighbors is not None:
            filename += f'_k{k_neighbors}'
        if distance_metric is not None:
            filename += f'_{distance_metric}'
        
        # Save plot
        fig_path = os.path.join(self.output_dir, f'{filename}.png')
        plt.savefig(fig_path)
        plt.close()
        self.logger.info(f"Plot saved to {fig_path}")
    
    def generate_comprehensive_plots(self, datasets: Optional[List[str]] = None) -> None:
        """
        Generate a comprehensive set of plots from the results.
        
        Parameters
        ----------
        datasets : list or None, optional
            List of datasets to plot. If None, use all datasets from results.
            
        Raises
        ------
        ValueError
            If no results are available.
        """
        if len(self.results) == 0:
            raise ValueError("No results available. Run comparison first.")
            
        if datasets is None:
            datasets = self.results['dataset'].unique()
        else:
            # Filter for datasets that exist in results
            datasets = [d for d in datasets if d in self.results['dataset'].unique()]
            
        if not datasets:
            self.logger.warning("No valid datasets specified for plotting.")
            return
            
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        # For each dataset and metric, plot metric vs test size for different k values
        for dataset in datasets:
            self.logger.info(f"Generating plots for dataset: {dataset}")
            for metric in metrics:
                if f'semi_{metric}' not in self.results.columns:
                    continue
                    
                try:
                    self.plot_metric_vs_test_size(metric, dataset, None)
                except Exception as e:
                    self.logger.error(f"Error plotting {metric} for {dataset}: {e}")
        
        # Also plot for specific k and different distance metrics
        for dataset in datasets:
            for metric in metrics:
                if f'semi_{metric}' not in self.results.columns:
                    continue
                    
                try:
                    # Select a mid-range k value
                    k_values = sorted(self.results['k_neighbors'].unique())
                    mid_k = k_values[len(k_values) // 2] if k_values else 5
                    self.plot_metric_vs_test_size(metric, dataset, mid_k, None)
                except Exception as e:
                    self.logger.error(f"Error plotting {metric} for {dataset} with k={mid_k}: {e}")


def run_default_benchmark():
    """
    Run a default benchmark with common datasets and configurations.
    
    Returns
    -------
    BenchmarkRunner
        The benchmark runner instance with results.
    """
    # Initialize the benchmark runner
    runner = BenchmarkRunner(output_dir='benchmark_results')
    
    # Define parameters
    datasets = ['iris', 'wine', 'breast_cancer']
    test_sizes = [0.3, 0.5, 0.7]
    k_values = [1, 3, 5, 7, 10]
    distance_metrics = ['euclidean', 'manhattan', 'cosine']
    
    # Run comparison
    runner.run_comparison(
        dataset_names=datasets,
        test_sizes=test_sizes,
        k_neighbors_values=k_values,
        distance_metrics=distance_metrics,
        max_depth=None
    )
    
    # Find best configurations
    best_configs = runner.find_best_configurations()
    
    # Generate plots
    runner.plot_best_improvement_heatmap(best_configs)
    runner.generate_comprehensive_plots()
    
    return runner


if __name__ == "__main__":
    run_default_benchmark() 