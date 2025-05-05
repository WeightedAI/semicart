# SemiCART: Semi-Supervised Decision Tree Algorithm

<div align="center">

<img src="results/accuracy-semi-cart.png" alt="SemiCART Performance" width="500"/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%E2%89%A50.24.0-orange.svg)](https://scikit-learn.org/)
[![Paper](https://img.shields.io/badge/paper-published-green.svg)](https://link.springer.com/article/10.1007/s13042-024-02161-z)
[![PyPI](https://img.shields.io/pypi/v/semicart.svg)](https://pypi.org/project/semicart/)
[![Downloads](https://img.shields.io/pypi/dm/semicart)](https://pypi.org/project/semicart/)

<strong>Building Semi-Supervised Decision Trees with Semi-CART Algorithm</strong>

</div>

SemiCART is a semi-supervised decision tree algorithm that enhances the traditional Classification and Regression Tree (CART) algorithm by incorporating semi-supervised learning principles. Published in the [International Journal of Machine Learning and Cybernetics](https://link.springer.com/article/10.1007/s13042-024-02161-z), our approach addresses a critical limitation of standard CART algorithms by leveraging unlabeled data in the training process.

## 🚀 Quick Links

- [Installation](#installation) - Get started with SemiCART
- [Quick Start](#quick-start) - Simple example to get you started
- [Benchmark Results](#benchmark-results-cart-vs-semicart) - See performance comparisons
- [How It Works](#how-it-works) - Learn about the algorithm
- [Examples](#examples) - Detailed usage examples

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [When to Use SemiCART](#when-to-use-semicart)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Performance Visualization](#performance-visualization)
- [Benchmark Results](#benchmark-results-cart-vs-semicart)
- [Parameters](#parameters)
- [Command-Line Interface](#command-line-interface)
- [Benchmarking](#benchmarking)
- [Examples](#examples)
- [Advantages](#advantages)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Contributing](#contributing)
- [Community](#community)
- [License](#license)

---

## Overview

Decision trees like CART form the foundation of modern boosting methodologies such as GBM, XGBoost, and LightGBM. However, standard CART algorithms can't learn from unlabeled data. SemiCART introduces "Distance-based Weighting," which leverages principles from graph-based semi-supervised learning to:

1. Calculate relevance of training records relative to test data
2. Remove irrelevant records to accelerate training
3. Improve overall performance through modified Gini index calculations

Our comprehensive evaluations across thirteen datasets from various domains demonstrate that SemiCART consistently outperforms standard CART methods, offering a significant contribution to statistical learning.

## Key Features

- **Distance-based Weighting**: Assigns weights to training instances based on their similarity to test instances, focusing the model on more relevant training data.
- **Modified Gini Index**: Incorporates instance weights into the splitting criteria, improving the decision tree's structure.
- **scikit-learn Compatible**: Fully compatible with the scikit-learn API, making it easy to integrate into existing ML pipelines.
- **Multiple Distance Metrics**: Supports a wide range of distance metrics (euclidean, manhattan, cosine, etc.)
- **Comprehensive Benchmarking**: Includes a benchmarking module for performance evaluation.
- **Cost-Effective Learning**: Efficiently utilizes both labeled and unlabeled data, reducing the need for expensive data labeling.

## When to Use SemiCART

SemiCART is particularly effective in scenarios where:

- You have limited labeled data but abundant unlabeled data
- There's a significant cost associated with data labeling
- You're working with datasets where traditional decision trees show high variance
- Your data comes from domains like medical diagnostics, fraud detection, or customer segmentation
- You require models with good interpretability, unlike black-box models
- You want to incorporate the structure of unlabeled data into your classification model

SemiCART's advantage increases with higher ratios of unlabeled to labeled data, making it ideal for semi-supervised learning tasks.

## Installation

### From PyPI (Recommended)

```bash
pip install semicart
```

### From Source (Latest Development Version)

```bash
git clone https://github.com/WeightedAI/semicart.git
cd semicart
pip install -e .
```

## Quick Start

```python
from semicart import SemiCART
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load and prepare data
X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Semi-CART model
model = SemiCART(k_neighbors=3, distance_metric='euclidean')
model.fit(X_train, y_train, X_test)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
# Output: Accuracy: 0.9778
```

### Distance-based Weighting

SemiCART introduces a novel approach to incorporate test data into the training phase, inspired by graph-based semi-supervised learning techniques:

1. For each test instance, distances to all training instances are calculated
2. The k-nearest training instances are identified for each test instance
3. Weights of these nearest training instances are incremented
4. Training instances with zero weight (not selected as neighbors) are removed
5. This focuses the model on the most relevant training data relative to the test set

### Modified Gini Index

SemiCART replaces the standard class proportions in the Gini index with weight-based proportions:

```
Modified Gini = 1 - Σ(w_i/S)²
```

Where:
- w_i = sum of weights of instances in class i
- S = total sum of weights in the subset

This modified splitting criterion ensures that the resulting decision tree better captures the underlying relationships between labeled and unlabeled data.

## Performance Visualization

SemiCART consistently outperforms traditional CART across various evaluation metrics:

### Accuracy Comparison
![Accuracy Comparison](results/accuracy-semi-cart.png)

### AUC Comparison
![AUC Comparison](results/auc-semi-cart.png)

### F1 Score Comparison
![F1 Score Comparison](results/f1-semi-cart.png)

These visualizations demonstrate SemiCART's superior performance across multiple datasets, particularly when leveraging unlabeled data effectively.

## Benchmark Results: CART vs SemiCART

Our extensive benchmarking across multiple datasets shows that SemiCART consistently outperforms traditional CART in classification tasks:

### Accuracy Improvements

| Dataset     | Test Size | k  | Best Distance Metric | CART   | SemiCART | Improvement |
|-------------|-----------|----|--------------------|--------|----------|-------------|
| banknote    | 0.1       | 2  | hamming            | 0.9928 | 1.0000   | +0.0072     |
| banknote    | 0.7       | 3  | yule               | 0.9594 | 0.9875   | +0.0281     |
| fertility   | 0.1       | 1  | jaccard            | 0.7000 | 0.9000   | +0.2000     |
| fertility   | 0.3       | 5  | jensenshannon      | 0.6333 | 0.8333   | +0.2000     |
| wdbc        | 0.1       | 3  | sqeuclidean        | 0.9298 | 1.0000   | +0.0702     |
| wdbc        | 0.3       | 7  | cosine             | 0.9006 | 0.9825   | +0.0819     |
| glass       | 0.1       | 18 | yule               | 0.6364 | 0.8636   | +0.2273     |
| glass       | 0.2       | 18 | sqeuclidean        | 0.7209 | 0.8837   | +0.1628     |
| transfusion | 0.1       | 6  | chebyshev          | 0.7067 | 0.7733   | +0.0667     |

### AUC Improvements

| Dataset     | Test Size | k  | Best Distance Metric | CART   | SemiCART | Improvement |
|-------------|-----------|----|--------------------|--------|----------|-------------|
| fertility   | 0.2       | 13 | jaccard            | 0.4722 | 0.9444   | +0.4722     |
| fertility   | 0.5       | 11 | jensenshannon      | 0.4545 | 0.7273   | +0.2727     |
| wdbc        | 0.1       | 3  | sqeuclidean        | 0.9147 | 1.0000   | +0.0853     |
| wdbc        | 0.3       | 7  | cosine             | 0.8892 | 0.9797   | +0.0905     |
| glass       | 0.1       | 3  | yule               | 0.8137 | 0.9386   | +0.1249     |
| glass       | 0.7       | 12 | hamming            | 0.7189 | 0.8346   | +0.1157     |

**Key observations**:
- SemiCART shows greatest improvements with smaller test sizes (more unlabeled data)
- Different distance metrics work best for different datasets
- Significant improvements even on datasets with complex decision boundaries
- Some datasets show dramatic improvements in AUC (up to +0.4722)

## Parameters

- `max_depth`: Maximum depth of the tree (default=None)
- `min_samples_split`: Minimum samples required to split a node (default=2)
- `k_neighbors`: Number of nearest neighbors to consider for weight assignment (default=1)
- `distance_metric`: Distance metric for similarity calculation (default='euclidean')
  - Supported values: 'euclidean', 'manhattan', 'cosine', 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'dice', 'hamming', 'jaccard', 'jensenshannon', 'minkowski', 'sqeuclidean', 'yule'
- `initial_weight`: Initial weight for each training instance (default=1.0)
- `weight_increment`: Weight increment for nearest neighbors (default=1.0)
- `random_state`: Random seed for reproducibility (default=None)
- `log_level`: Logging level (default=logging.INFO)

## Command-Line Interface

SemiCART includes a convenient command-line interface for quick experimentation:

```bash
# Run with default parameters on Iris dataset
semicart

# Run with custom parameters
semicart --dataset wine --test-size 0.4 --k-neighbors 5 --distance-metric manhattan

# Get help on available options
semicart --help
```

## Benchmarking

SemiCART includes a comprehensive benchmarking module for evaluating performance:

```python
from semicart.benchmark import run_default_benchmark

# Run a default benchmark on common datasets
runner = run_default_benchmark()

# Or create a custom benchmark
from semicart.benchmark import BenchmarkRunner

runner = BenchmarkRunner(output_dir='my_results')
runner.run_comparison(
    dataset_names=['iris', 'wine'],
    test_sizes=[0.3, 0.5],
    k_neighbors_values=[1, 3, 5],
    distance_metrics=['euclidean', 'manhattan']
)
```

## Examples

Check out the `examples` directory for more detailed usage examples:

- `simple_example.py`: Basic comparison with standard CART
- `distance_metrics_comparison.py`: Comparing different distance metrics
- `advanced_usage.py`: More advanced options and configurations

### Basic Comparison with scikit-learn's DecisionTreeClassifier

```python
from semicart import SemiCART
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load and prepare data
X, y = load_wine(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train standard CART
cart = DecisionTreeClassifier(random_state=42)
cart.fit(X_train, y_train)
cart_pred = cart.predict(X_test)
cart_acc = accuracy_score(y_test, cart_pred)

# Train SemiCART
semicart = SemiCART(k_neighbors=5, distance_metric='euclidean', random_state=42)
semicart.fit(X_train, y_train, X_test)
semicart_pred = semicart.predict(X_test)
semicart_acc = accuracy_score(y_test, semicart_pred)

print(f"CART Accuracy: {cart_acc:.4f}")
print(f"SemiCART Accuracy: {semicart_acc:.4f}")
print(f"Improvement: {semicart_acc - cart_acc:.4f}")
```

## Advantages

- **Improved Accuracy**: SemiCART consistently outperforms CART on a wide range of datasets
- **Utilizes Unlabeled Data**: Leverages unlabeled instances to enhance the learning process
- **Cost-Effective**: Reduces the need for extensive data labeling
- **Flexibility**: Works with various distance metrics to adapt to different data distributions
- **Interpretability**: Maintains the interpretability of decision trees
- **Integration**: Easily integrates into existing ML pipelines through scikit-learn compatibility
- **Domain-Agnostic**: Performs well across various domains and data types

## Requirements

SemiCART requires the following dependencies:

- Python ≥ 3.7
- NumPy ≥ 1.19.0
- scikit-learn ≥ 0.24.0
- SciPy ≥ 1.6.0
- pandas ≥ 1.0.0

Compatible with all major operating systems (Windows, macOS, Linux).

## Troubleshooting

### Common Issues

**ImportError: No module named 'semicart'**
- Make sure you've installed the package with `pip install semicart`
- Verify your Python environment is activated if using virtual environments

**AttributeError when using SemiCART with custom datasets**
- Ensure your data is properly formatted (numerical, no NaN values)
- Check that feature scaling is applied for distance-based metrics

**Poor performance on specific datasets**
- Try different distance metrics (results vary by dataset characteristics)
- Adjust the k_neighbors parameter (often 3-7 works well for most datasets)
- Ensure proper feature scaling is applied

For more help, please [open an issue](https://github.com/WeightedAI/semicart/issues) on our GitHub repository.

## Citation

If you use SemiCART in your research, please cite the following paper:

```
@article{abedinia2024semicart,
  title={Building Semi-Supervised Decision Trees with Semi-CART Algorithm},
  author={Abedinia, Aydin and Seydi, Vahid},
  journal={International Journal of Machine Learning and Cybernetics},
  volume={15},
  pages={4493--4510},
  year={2024},
  publisher={Springer},
  doi={10.1007/s13042-024-02161-z}
}
```

## Contributing

Contributions to SemiCART are welcome! Please check our [contributing guidelines](CONTRIBUTING.md) for more details.

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a virtual environment and install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Create a branch for your changes
5. Make your changes and add tests
6. Run tests locally:
   ```bash
   pytest
   ```
7. Submit a pull request

## Community

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For usage questions and discussions
- **Pull Requests**: For contributing code and documentation

Join our community of data scientists and machine learning practitioners to improve SemiCART and expand its capabilities!

## License

SemiCART is released under the MIT License. See the [LICENSE](LICENSE) file for details.
