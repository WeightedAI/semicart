import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from db_weights.weights import WeightCalculator


class DecisionNode:
    def __init__(self, feature_i=None, threshold=None, value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch


class SemiCARTClassifier:
    def __init__(self, weights, strategy="GINI", min_samples_split=2, min_impurity=1e-7, max_depth=float("inf")):
        self.root = None
        self.one_dim = False
        self._impurity_calculation = None
        self._leaf_value_calculation = None

        self.weights = weights
        self.max_depth = max_depth
        self.min_impurity = min_impurity
        self.min_samples_split = min_samples_split

        if strategy == "GINI":
            self._impurity_calculation = self._gini_calculate_information_gain
        elif strategy == "ENTROPY":
            self._impurity_calculation = self._entropy_calculate_information_gain

        self._leaf_value_calculation = self._majority_vote

    def predict(self, X):
        return [self.predict_value(sample, self.root) for sample in X]

    def predict_value(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_value = x[tree.feature_i]
        branch = tree.false_branch if feature_value < tree.threshold else tree.true_branch
        return self.predict_value(x, branch)

    def _majority_vote(self, y):
        most_common = np.bincount(y).argmax()
        return most_common

    def fit(self, X, y):
        self.one_dim = len(y.shape) == 1
        self.root = self._build_tree(X, y, 0, self.weights)

    def _divide_on_feature(self, X, feature_i, threshold, weights):
        split_func = X[:, feature_i] >= threshold

        X_left = X[split_func]
        X_right = X[~split_func]

        weights_left = weights[split_func]
        weights_right = weights[~split_func]

        return X_left, X_right, weights_left, weights_right

    def _build_tree(self, X, y, current_depth=0, weights=[]):
        largest_impurity = 0
        best_criteria = None
        best_sets = None

        if len(X) >= self.min_samples_split and current_depth <= self.max_depth:
            for feature_i in range(X.shape[1]):
                unique_values = np.unique(X[:, feature_i])
                for threshold in unique_values:

                    X_left, X_right, weights_left, weights_right = self._divide_on_feature(
                        X, feature_i, threshold, weights
                    )

                    if len(X_left) > 0 and len(X_right) > 0:
                        y_left, y_right = y[X[:, feature_i] >= threshold], y[X[:, feature_i] < threshold]
                        impurity = self._impurity_calculation(
                            y,
                            y_left,
                            y_right,
                            weights,
                            weights_left,
                            weights_right
                        )

                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": X_left, "lefty": y_left,
                                "rightX": X_right, "righty": y_right,
                                "weights_left": weights_left, "weights_right": weights_right
                            }

        if largest_impurity > self.min_impurity:
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1,
                                           best_sets["weights_left"])
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1,
                                            best_sets["weights_right"])
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria["threshold"],
                                true_branch=true_branch, false_branch=false_branch)
        leaf_value = self._majority_vote(y)
        return DecisionNode(value=leaf_value)

    def _gini_calculate_information_gain(self, y, y_left, y_right, weights, weights_left, weights_right):
        original_impurity = self._calculate_gini_impurity(y, weights)
        impurity_left = self._calculate_gini_impurity(y_left, weights_left)
        impurity_right = self._calculate_gini_impurity(y_right, weights_right)

        total_weight = np.sum(weights)
        total_weight_left = np.sum(weights_left)
        total_weight_right = np.sum(weights_right)

        p_left = total_weight_left / total_weight
        p_right = total_weight_right / total_weight

        weighted_impurity = p_left * impurity_left + p_right * impurity_right

        info_gain = original_impurity - weighted_impurity

        return info_gain

    def _calculate_gini_impurity(self, y, weights):
        unique_labels, counts = np.unique(y, return_counts=True)
        total_weight = np.sum(weights)

        weighted_counts = np.zeros_like(unique_labels, dtype=float)
        for i, label in enumerate(unique_labels):
            weighted_counts[i] = np.sum(weights[y == label])

        probabilities = weighted_counts / total_weight
        gini_impurity = 1 - np.sum(probabilities ** 2)

        return gini_impurity

    def _calculate_weighted_entropy(self, y, weights):
        total_weight = np.sum(weights)
        unique_labels, counts = np.unique(y, return_counts=True)
        weighted_counts = np.zeros_like(unique_labels, dtype=float)

        for i, label in enumerate(unique_labels):
            weighted_counts[i] = np.sum(weights[y == label])

        probabilities = weighted_counts / total_weight
        entropy = 1 - np.sum(probabilities * np.log2(probabilities))
        return entropy

    def _entropy_calculate_information_gain(self, y, y_left, y_right, weights, weights_left, weights_right):
        original_entropy = self._calculate_weighted_entropy(y, weights)
        entropy_left = self._calculate_weighted_entropy(y_left, weights_left)
        entropy_right = self._calculate_weighted_entropy(y_right, weights_right)

        total_weight = np.sum(weights)
        total_weight_left = np.sum(weights_left)
        total_weight_right = np.sum(weights_right)

        p_left = total_weight_left / total_weight
        p_right = total_weight_right / total_weight

        weighted_entropy = p_left * entropy_left + p_right * entropy_right

        info_gain = original_entropy - weighted_entropy

        return info_gain


def tuning_params(X_train, X_test, y_train, y_test, neighbors) -> dict:
    result = {}
    strategies = ["ENTROPY", "GINI"]
    best_score = 0
    if neighbors is None:
        neighbors = [1, 2, 3, 4, 5]

    def max_score(result):
        return max(result["accuracy_score"], result["precision_score"], result["recall_score"], result["f1_score"])

    for n in tqdm(neighbors, desc="Neighbors"):
        weights_calculator = WeightCalculator()
        weights = weights_calculator.calculate_weights_nn(X_train, X_test, n)

        for strategy_param in tqdm(strategies, desc="Strategies"):
            tree = SemiCARTClassifier(weights, strategy=strategy_param)
            tree.fit(X_train, y_train)
            y_pred = tree.predict(X_test)

            result["knn"] = {
                "method": strategy_param,
                "num neighbors": n,
                "accuracy_score": accuracy_score(y_test, y_pred),
                "precision_score": precision_score(y_test, y_pred),
                "recall_score": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
            }

            if max_score(result["knn"]) > best_score:
                best_score = max_score(result["knn"])
                best_result = result["knn"]
                print(f"New best result: {best_result}")

            for measure in tqdm(weights_calculator.get_measurements(), desc="distance measurements"):
                weights = weights_calculator.calculate_weights_dist(X_train, X_test, n, measure)
                tree = SemiCARTClassifier(weights, strategy_param)
                tree.fit(X_train, y_train)
                y_pred = tree.predict(X_test)

                result[measure] = {
                    "method": strategy_param,
                    "num neighbors": n,
                    "accuracy_score": accuracy_score(y_test, y_pred),
                    "precision_score": precision_score(y_test, y_pred),
                    "recall_score": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                }

                if max_score(result[measure]) > best_score:
                    best_score = max_score(result[measure])
                    best_result = result[measure]
                    print(f"New best result: {best_result}")

    return result
