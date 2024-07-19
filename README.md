# SemiCart

SemiCart is an algorithm based on the Classification and Regression Trees (CART) that utilizes the weights of test data to enhance prediction accuracy. This algorithm employs methods such as Nearest Neighbor and metrics like Euclidean and Mahalanobis distances to determine these weights.

## Features

- Semi-supervised decision tree algorithm.
- Utilizes Nearest Neighbor and distance metrics for weight calculations.
- Enhances prediction accuracy by considering test data weights.

## Installation

You can install SemiCart via pip:

```bash
pip install semicart
```

```bash
git clone https://github.com/WeightedBasedAI/semicart.git
cd semicart
python setup.py install
```

## Usage
Here is an example of how to use SemiCart:

```python
from semicart import WeightCalculator, SemiCARTClassifier

# Calculate weights using Nearest Neighbor
weights_calculator = WeightCalculator()
weights = weights_calculator.calculate_weights_nn(X_train, X_test, n)

# Create and train the SemiCARTClassifier
tree = SemiCARTClassifier(weights, strategy=strategy_param)
tree.fit(X_train, y_train)

# Predict using the trained classifier
y_pred = tree.predict(X_test)

# Calculate weights using distance metrics (Euclidean or Mahalanobis)
weights = weights_calculator.calculate_weights_dist(X_train, X_test, n, measure)

# Create and train the SemiCARTClassifier with distance metrics
tree = SemiCARTClassifier(weights, strategy_param)
tree.fit(X_train, y_train)

# Predict using the trained classifier
y_pred = tree.predict(X_test)

# Tuning the SemiCart with loading progress on neighbors, strategy, number of neighbors
results = tuning_params(X_train, X_test, y_train, y_test)
```

## Testing
To run tests, use the following command:

```python
python -m unittest discover -s tests
```

## Building the Package
To create a wheel package, run:

```python
python setup.py sdist bdist_wheel
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
## License

This project is licensed under the MIT License.
## Author

Aydin Abedinia - Vahid Seydi

## Acknowledgments
For more information, please refer to the Springer article.
https://link.springer.com/article/10.1007/s13042-024-02161-z