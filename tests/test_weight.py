# test_weight.py

import unittest
import numpy as np
from semicart.weights import WeightCalculator


class TestWeightCalculator(unittest.TestCase):

    def setUp(self):
        self.calculator = WeightCalculator(n_neighbors=3)
        self.x_train = np.array([[0, 0], [1, 1], [2, 2]])
        self.x_test = np.array([[1, 0], [2, 1]])

    def test_calculate_weights_nn(self):
        weights = self.calculator.calculate_weights_nn(self.x_train, self.x_test)
        self.assertEqual(len(weights), len(self.x_train))
        self.assertTrue(all(weights > 1))


if __name__ == '__main__':
    unittest.main()
