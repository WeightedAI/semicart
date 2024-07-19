import unittest
import numpy as np
from semicart import SemiCARTClassifier, tuning_params
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class TestSemiCARTClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_classes=2, random_state=42)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def test_majority_vote(self):
        tree = SemiCARTClassifier(weights=np.ones(len(self.y_train)))
        y = np.array([0, 0, 1, 1, 1])
        self.assertEqual(tree._majority_vote(y), 1)

    def test_fit_and_predict(self):
        tree = SemiCARTClassifier(weights=np.ones(len(self.X_train)), strategy="GINI")
        tree.fit(self.X_train, self.y_train)
        y_pred = tree.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertGreaterEqual(accuracy, 0.5)

    def test_gini_impurity(self):
        tree = SemiCARTClassifier(weights=np.ones(len(self.y_train)))
        y = np.array([0, 0, 1, 1])
        weights = np.ones(len(y))
        gini = tree._calculate_gini_impurity(y, weights)
        self.assertAlmostEqual(gini, 0.5, places=1)


class TestTuningParams(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_classes=2, random_state=42)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def test_tuning_params(self):
        result = tuning_params(self.X_train, self.X_test, self.y_train, self.y_test, neighbors=[1, 2])
        self.assertIn('accuracy_score', result['knn'])
        self.assertIn('precision_score', result['knn'])
        self.assertIn('recall_score', result['knn'])
        self.assertIn('f1_score', result['knn'])
        self.assertGreater(result['knn']['accuracy_score'], 0.5)


if __name__ == "__main__":
    unittest.main()
