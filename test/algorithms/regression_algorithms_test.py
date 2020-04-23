import unittest

from src.algorithms.regression_algorithms import stochastic_gradient_descent_coefficients, \
    predict_with_multivariate_linear_regression


class MyTestCase(unittest.TestCase):
    def test_stochastic_gradient_descent_coefficient_calculation(self):
        dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
        learning_rate = 0.001
        num_of_epochs = 50

        actual_coefficients = stochastic_gradient_descent_coefficients(dataset, learning_rate, num_of_epochs)

        expected_coefficients = [0.22998, 0.80172]
        for index in range(len(actual_coefficients)):
            self.assertAlmostEqual(
                actual_coefficients[index],
                expected_coefficients[index],
                delta=0.00001
            )

    def test_multivariate_linear_regression_prediction(self):
        dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
        coefficients = [0.4, 0.8]

        actual_predictions = list()
        for row in dataset:
            actual_predictions.append(
                predict_with_multivariate_linear_regression(row, coefficients)
            )

        expected_predictions = [1.2, 2, 3.6, 2.8, 4.4]
        for index in range(len(actual_predictions)):
            self.assertAlmostEqual(
                actual_predictions[index],
                expected_predictions[index],
                delta=0.1
            )


if __name__ == '__main__':
    unittest.main()
