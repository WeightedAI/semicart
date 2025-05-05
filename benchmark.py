import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import (
    euclidean_distances, manhattan_distances, cosine_distances
)
from scipy.spatial.distance import (
    braycurtis, canberra, chebyshev, cityblock, correlation, 
    dice, euclidean, hamming, jaccard, jensenshannon, 
    minkowski, sqeuclidean, 
    yule
)
import time
import warnings
warnings.filterwarnings('ignore')

from semicart import SemiCART

# Dictionary of distance functions
DISTANCE_FUNCS = {
    'euclidean': euclidean_distances,
    'manhattan': manhattan_distances,
    'cosine': cosine_distances,
    'braycurtis': lambda X, Y: np.array([[braycurtis(x, y) for y in Y] for x in X]),
    'canberra': lambda X, Y: np.array([[canberra(x, y) for y in Y] for x in X]),
    'chebyshev': lambda X, Y: np.array([[chebyshev(x, y) for y in Y] for x in X]),
    'cityblock': lambda X, Y: np.array([[cityblock(x, y) for y in Y] for x in X]),
    'correlation': lambda X, Y: np.array([[correlation(x, y) for y in Y] for x in X]),
    'dice': lambda X, Y: np.array([[dice(x, y) for y in Y] for x in X]),
    'hamming': lambda X, Y: np.array([[hamming(x, y) for y in Y] for x in X]),
    'jaccard': lambda X, Y: np.array([[jaccard(x, y) for y in Y] for x in X]),
    'jensenshannon': lambda X, Y: np.array([[jensenshannon(x, y) for y in Y] for x in X]),
    'minkowski': lambda X, Y: np.array([[minkowski(x, y, 3) for y in Y] for x in X]),
    'sqeuclidean': lambda X, Y: np.array([[sqeuclidean(x, y) for y in Y] for x in X]),
    'yule': lambda X, Y: np.array([[yule(x, y) for y in Y] for x in X]),
}

# We'll use all available distance metrics for testing
SELECTED_DISTANCES = list(DISTANCE_FUNCS.keys())

def get_dataset(name):
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
        print(f"Error loading dataset {name}: {e}")
        raise

def evaluate_model(y_true, y_pred, y_score=None, average='macro'):
    """
    Evaluate a model using multiple metrics.
    
    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    y_score : array-like, optional
        Probability estimates, required for AUC.
    average : str, optional
        Averaging strategy for multiclass.
        
    Returns
    -------
    dict
        Dictionary of evaluation metrics.
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
        except:
            metrics['auc'] = np.nan
    
    return metrics

def plot_metric_vs_test_size(results, metric, dataset_name, k_neighbors, distance_metric=None):
    """
    Plot a specific metric vs test size.
    
    Parameters
    ----------
    results : DataFrame
        Results dataframe.
    metric : str
        Metric to plot.
    dataset_name : str
        Name of the dataset.
    k_neighbors : int or None
        Number of neighbors (if None, will plot for all).
    distance_metric : str or None
        Distance metric (if None, will use euclidean).
    """
    plt.figure(figsize=(12, 8))
    
    # Filter results
    if k_neighbors is not None:
        filtered_results = results[(results['dataset'] == dataset_name) & 
                                  (results['k_neighbors'] == k_neighbors)]
        if distance_metric is not None:
            filtered_results = filtered_results[filtered_results['distance_metric'] == distance_metric]
    else:
        filtered_results = results[results['dataset'] == dataset_name]
    
    # Prepare data for plotting
    test_sizes = sorted(filtered_results['test_size'].unique())
    
    if k_neighbors is None:
        # Plot for all k values with euclidean distance
        k_values = sorted(filtered_results['k_neighbors'].unique())
        for k in k_values:
            k_data = filtered_results[(filtered_results['k_neighbors'] == k) & 
                                     (filtered_results['distance_metric'] == 'euclidean')]
            
            semi_values = [k_data[k_data['test_size'] == ts][f'semi_{metric}'].values[0] 
                          for ts in test_sizes]
            plt.plot(test_sizes, semi_values, marker='o', label=f'Semi-CART k={k}')
    else:
        # Plot for specific k with different distance metrics or just euclidean
        if distance_metric is None:
            distance_metrics = sorted(filtered_results['distance_metric'].unique())
            for dm in distance_metrics:
                dm_data = filtered_results[filtered_results['distance_metric'] == dm]
                semi_values = [dm_data[dm_data['test_size'] == ts][f'semi_{metric}'].values[0] 
                              for ts in test_sizes]
                plt.plot(test_sizes, semi_values, marker='o', label=f'Semi-CART {dm}')
        else:
            # Plot for specific k and distance metric
            semi_values = [filtered_results[filtered_results['test_size'] == ts][f'semi_{metric}'].values[0] 
                          for ts in test_sizes]
            plt.plot(test_sizes, semi_values, marker='o', label=f'Semi-CART k={k_neighbors}, {distance_metric}')
    
    # Always plot standard CART for comparison
    standard_values = [filtered_results[filtered_results['test_size'] == ts][f'cart_{metric}'].values[0] 
                      for ts in test_sizes]
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
    
    filename = f'{dataset_name.lower()}_{metric}_vs_testsize'
    if k_neighbors is not None:
        filename += f'_k{k_neighbors}'
    if distance_metric is not None:
        filename += f'_{distance_metric}'
    
    plt.savefig(f'./output/{filename}.png') 
    plt.close()

def run_comprehensive_comparison(dataset_names, test_sizes, k_neighbors_values, distance_metrics):
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
        
    Returns
    -------
    DataFrame
        Results dataframe.
    """
    results = []
    
    for dataset_name in dataset_names:
        print(f"\nProcessing dataset: {dataset_name}")
        
        try:
            # Load dataset
            X, y = get_dataset(dataset_name)
            
            # Handle empty or invalid datasets
            if X is None or y is None or len(X) == 0 or len(y) == 0:
                print(f"  Warning: Empty or invalid dataset for {dataset_name}, skipping.")
                continue
                
            # Check if X contains NaN or infinite values
            if np.isnan(X).any() or np.isinf(X).any():
                print(f"  Warning: Dataset {dataset_name} contains NaN or infinite values. Applying preprocessing.")
                # Replace NaN with 0 and infinite with large values
                X = np.nan_to_num(X)
            
            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            for test_size in test_sizes:
                print(f"  Test size: {test_size:.1f}")
                
                try:
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
                    )
                    
                    # Train and evaluate standard CART
                    start_time = time.time()
                    cart = DecisionTreeClassifier(random_state=42)
                    cart.fit(X_train, y_train)
                    cart_time = time.time() - start_time
                    
                    y_pred_cart = cart.predict(X_test)
                    try:
                        y_proba_cart = cart.predict_proba(X_test)
                    except:
                        y_proba_cart = None
                    
                    cart_metrics = evaluate_model(y_test, y_pred_cart, y_proba_cart)
                    
                    # Try different k_neighbors values and distance metrics
                    for k in k_neighbors_values:
                        for dist_metric in distance_metrics:
                            try:
                                print(f"    k={k}, distance={dist_metric}", end='\r')
                                
                                # Train and evaluate Semi-CART
                                start_time = time.time()
                                semicart = SemiCART(k_neighbors=k, distance_metric=dist_metric)
                                semicart.fit(X_train, y_train, X_test)
                                semicart_time = time.time() - start_time
                                
                                y_pred_semi = semicart.predict(X_test)
                                try:
                                    y_proba_semi = semicart.predict_proba(X_test)
                                except:
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
                                        print(f"    k={k}, distance={dist_metric}, accuracy improvement: +{acc_improvement:.4f}")
                                
                            except Exception as e:
                                print(f"    Error with k={k}, distance={dist_metric}: {e}")
                                continue
                
                except Exception as e:
                    print(f"  Error processing test_size={test_size} for dataset {dataset_name}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            continue
    
    # Convert to DataFrame
    if not results:
        print("No results were collected. Check error messages above.")
        return pd.DataFrame()
        
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv('./output/comprehensive_comparison_results.csv', index=False)
    
    print("\nComprehensive comparison completed!")
    return results_df

def find_best_configurations(results):
    """
    Find the best configurations across datasets and test sizes.
    
    Parameters
    ----------
    results : DataFrame
        Results dataframe.
        
    Returns
    -------
    DataFrame
        Best configurations dataframe.
    """
    best_configs = []
    
    for dataset in results['dataset'].unique():
        for test_size in results['test_size'].unique():
            dataset_size_results = results[(results['dataset'] == dataset) & 
                                           (results['test_size'] == test_size)]
            
            # Find best configuration for each metric
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
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
                except:
                    continue
    
    best_configs_df = pd.DataFrame(best_configs)
    best_configs_df = best_configs_df[[
        'dataset', 'test_size', 'metric', 'k_neighbors', 'distance_metric',
        'cart_value', 'semi_value', 'improvement'
    ]]
    
    # Save best configurations to CSV
    best_configs_df.to_csv('./output/best_configurations.csv', index=False)
    
    return best_configs_df

def plot_best_improvement_heatmap(best_configs):
    """
    Plot a heatmap of the best improvements for each dataset and test size.
    
    Parameters
    ----------
    best_configs : DataFrame
        Best configurations dataframe.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    for metric in metrics:
        # Filter for the specific metric
        metric_data = best_configs[best_configs['metric'] == metric]
        
        # Create pivot table for heatmap
        pivot_data = metric_data.pivot(index='dataset', columns='test_size', values='improvement')
        
        # Plot heatmap
        plt.figure(figsize=(14, 8))
        ax = sns.heatmap(pivot_data, annot=True, cmap='RdBu_r', center=0, fmt='.4f')
        plt.title(f'Best Semi-CART Improvement for {metric.capitalize()}', fontsize=16)
        plt.xlabel('Test Size', fontsize=14)
        plt.ylabel('Dataset', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'./output/best_improvement_{metric}_heatmap.png')
        plt.close()

def generate_comprehensive_plots(results):
    """
    Generate a comprehensive set of plots from the results.
    
    Parameters
    ----------
    results : DataFrame
        Results dataframe.
    """
    datasets = results['dataset'].unique()
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # For each dataset and metric, plot metric vs test size for different k values
    for dataset in datasets:
        for metric in metrics:
            plot_metric_vs_test_size(results, metric, dataset, None)
    
    # Also plot for specific k and different distance metrics
    for dataset in datasets:
        for metric in metrics:
            # Select a mid-range k value
            plot_metric_vs_test_size(results, metric, dataset, 5, None)

def plot_training_data_size_effect(results):
    """
    Plot the effect of training data size on model performance.
    
    Parameters
    ----------
    results : DataFrame
        Results dataframe.
    """
    datasets = results['dataset'].unique()
    
    for dataset in datasets:
        plt.figure(figsize=(14, 8))
        
        # Filter results for this dataset and Euclidean distance with k=5
        filtered_results = results[(results['dataset'] == dataset) & 
                                  (results['k_neighbors'] == 5) &
                                  (results['distance_metric'] == 'euclidean')]
        
        # Calculate training set size as 1 - test_size
        filtered_results['train_size'] = 1 - filtered_results['test_size']
        
        # Sort by training size
        filtered_results = filtered_results.sort_values('train_size')
        
        # Plot for accuracy
        plt.plot(filtered_results['train_size'], filtered_results['cart_accuracy'], 
                marker='s', linestyle='--', color='red', linewidth=2, label='Standard CART')
        plt.plot(filtered_results['train_size'], filtered_results['semi_accuracy'], 
                marker='o', color='blue', linewidth=2, label='Semi-CART (k=5)')
        
        plt.title(f'Accuracy vs Training Data Size - {dataset} Dataset', fontsize=16)
        plt.xlabel('Training Data Size (proportion)', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Add improvement annotations
        for i, row in filtered_results.iterrows():
            improvement = row['improvement_accuracy']
            plt.text(row['train_size'], max(row['cart_accuracy'], row['semi_accuracy']) + 0.01,
                   f'{improvement:.4f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'./output/{dataset.lower()}_accuracy_vs_trainsize.png')
        plt.close()

def generate_latex_table(best_configs):
    """
    Generate a LaTeX table from the best configurations.
    
    Parameters
    ----------
    best_configs : DataFrame
        Best configurations dataframe.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    datasets = best_configs['dataset'].unique()
    
    for metric in metrics:
        # Start LaTeX table
        latex_table = "\\begin{table}[ht]\n"
        latex_table += "\\centering\n"
        latex_table += f"\\caption{{Best Semi-CART configurations for {metric.capitalize()}}}\n"
        latex_table += "\\begin{tabular}{lcccccc}\n"
        latex_table += "\\hline\n"
        latex_table += "Dataset & Test Size & k & Distance & CART & Semi-CART & Improvement \\\\ \\hline\n"
        
        for dataset in datasets:
            for test_size in sorted(best_configs['test_size'].unique()):
                # Get row for this dataset, test size, and metric
                row = best_configs[
                    (best_configs['dataset'] == dataset) & 
                    (best_configs['test_size'] == test_size) &
                    (best_configs['metric'] == metric)
                ]
                
                if not row.empty:
                    row = row.iloc[0]
                    latex_table += f"{dataset} & {test_size:.1f} & {int(row['k_neighbors'])} & "
                    latex_table += f"{row['distance_metric']} & {row['cart_value']:.4f} & "
                    latex_table += f"{row['semi_value']:.4f} & {row['improvement']:.4f} \\\\\n"
        
        latex_table += "\\hline\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\end{table}"
        
        # Save to file
        with open(f'./output/latex_table_{metric}.tex', 'w') as f:
            f.write(latex_table)

def run_analysis():
    """
    Run the full analysis.
    """
    # Parameters
    dataset_names = ['banknote', 'fertility', 'wdbc', 'biodeg', 'haberman', 
                     'transfusion', 'hepatitis', 'tictactoe', 'vote', 'bupa', 
                     'breast', 'glass', 'mammographic_masses']
    
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    k_neighbors_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    # Use all available distance metrics
    distance_metrics = SELECTED_DISTANCES
    
    print(f"Starting comprehensive comparison with:")
    print(f"  - {len(dataset_names)} datasets: {', '.join(dataset_names)}")
    print(f"  - {len(test_sizes)} test sizes: {', '.join([str(ts) for ts in test_sizes])}")
    print(f"  - {len(k_neighbors_values)} k values: {', '.join([str(k) for k in k_neighbors_values])}")
    print(f"  - {len(distance_metrics)} distance metrics: {', '.join(distance_metrics)}")
    print("\nThis will result in", 
          len(dataset_names) * len(test_sizes) * len(k_neighbors_values) * len(distance_metrics),
          "total model evaluations.")
    
    # Run comparison
    print("\nStarting comparison...")
    results = run_comprehensive_comparison(
        dataset_names, test_sizes, k_neighbors_values, distance_metrics
    )
    
    # If no results were collected, exit
    if results.empty:
        print("No results to analyze. Exiting.")
        return
    
    # Find best configurations
    print("\nFinding best configurations...")
    best_configs = find_best_configurations(results)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_best_improvement_heatmap(best_configs)
    
    # Select a few representative datasets for detailed plots to avoid overloading
    representative_datasets = list(results['dataset'].unique())[:5]
    detailed_results = results[results['dataset'].isin(representative_datasets)]
    
    generate_comprehensive_plots(detailed_results)
    plot_training_data_size_effect(detailed_results)
    
    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    generate_latex_table(best_configs)
    
    print("\nAnalysis complete!")
    print("All results have been saved to CSV files and plots.")
    
    # Print summary of best improvements
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    print("\nSummary of best improvements for each dataset:")
    
    for dataset in best_configs['dataset'].unique():
        print(f"\n{dataset.upper()}")
        for metric in metrics:
            dataset_metric_configs = best_configs[
                (best_configs['dataset'] == dataset) & 
                (best_configs['metric'] == metric)
            ]
            
            if not dataset_metric_configs.empty:
                best_row = dataset_metric_configs.iloc[0]
                print(f"  {metric.upper()}: {best_row['improvement']:.4f} improvement with k={int(best_row['k_neighbors'])}, "
                     f"distance={best_row['distance_metric']}, test_size={best_row['test_size']:.1f}")
    
    return results, best_configs

if __name__ == "__main__":
    run_analysis()
