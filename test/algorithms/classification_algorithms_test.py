import unittest

from src.algorithms.classification_algorithms import predict_with_multivariate_linear_regression


class MyTestCase(unittest.TestCase):
    def test_multivariate_linear_regression(self):
        dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
        coefficients = [0.4, 0.8]

        actual_predictions = list()
        for row in dataset:
            actual_predictions.append(
                predict_with_multivariate_linear_regression(row, coefficients)
            )

        expected_predictions = [1.2, 2, 3.6, 2.8, 4.4]
        for index in range(len(actual_predictions)):
            self.assertAlmostEquals(
                actual_predictions[index],
                expected_predictions[index],
                delta=0.0001
            )


if __name__ == '__main__':
    unittest.main()
