# SemiCart

SemiCart is an algorithm based on the Classification and Regression Trees (CART) that utilizes the weights of test data to enhance prediction accuracy. This algorithm employs methods such as Nearest Neighbor and metrics like Euclidean and Mahalanobis distances to determine these weights.


https://pypi.org/project/semicart/


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
from semicart import SemiCARTClassifier
from db_weights.weights import WeightCalculator

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

# Example
```python
!pip install semicart
!pip install db_weights

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from db_weights import WeightCalculator
from semicart.semicart import SemiCARTClassifier, tuning_params


df = pd.read_csv(DATASET_PATH)
X = np.array(df.iloc[:, :-1].values.tolist())
y = np.array(df.iloc[:, -1].values.tolist())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# find best parameters on semi-supervised approach, measurements tuning with num of neighbors and GINI or ENTROPY
best_weights = tuning_params(X_train, X_test, y_train, y_test, [1,2,3,4,5,6,7,8,8,10])

# train with weights calculated from nearest neighbors 
weights = WeightCalculator().calculate_weights_nn(X_train, X_test, weight=1)

tree = SemiCARTClassifier(weights=weights,  strategy="GINI")
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

print("accuracy: ", accuracy_score(y_test, y_pred))
print("precision: ", precision_score(y_test, y_pred))
print("recall: ", recall_score(y_test, y_pred))
print("f1: ", f1_score(y_test, y_pred))
# accuracy:  0.6231884057971014
# precision:  0.5161290322580645
# recall:  0.5925925925925926
# f1:  0.5517241379310345

# trained with distance measurements which tuned before get params like 2, braycurtis, and GINI
weights = WeightCalculator().calculate_weights_dist(X_train, X_test, weight=2, measure_type='braycurtis')

tree = SemiCARTClassifier(weights=weights,  strategy="GINI")
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

print("accuracy: ", accuracy_score(y_test, y_pred))
print("precision: ", precision_score(y_test, y_pred))
print("recall: ", recall_score(y_test, y_pred))
print("f1: ", f1_score(y_test, y_pred))

# accuracy:  0.7681159420289855
# precision:  0.7037037037037037
# recall:  0.7037037037037037
# f1:  0.7037037037037037

# then compare to decision trees from scikit
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("accuracy: ", accuracy_score(y_test, y_pred))
print("precision: ", precision_score(y_test, y_pred))
print("recall: ", recall_score(y_test, y_pred))
print("f1: ", f1_score(y_test, y_pred))


# accuracy:  0.6811594202898551
# precision:  0.5925925925925926
# recall:  0.5925925925925926
# f1:  0.5925925925925926

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

