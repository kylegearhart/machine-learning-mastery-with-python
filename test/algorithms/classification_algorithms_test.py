import unittest

from src.algorithms.classification_algorithms import predict_with_logistic_regression_classification, \
    update_coefficients_with_logistic_regression, predict_with_single_perceptron_classification, \
    update_weights_with_single_perceptron
from src.algorithms.stochastic_gradient_descent import stochastic_gradient_descent


class MyTestCase(unittest.TestCase):
    def test_multivariate_logistic_classification_prediction(self):
        dataset = [[2.7810836, 2.550537003, 0], [1.465489372, 2.362125076, 0], [3.396561688, 4.400293529, 0],
                   [1.38807019, 1.850220317, 0], [3.06407232, 3.005305973, 0], [7.627531214, 2.759262235, 1],
                   [5.332441248, 2.088626775, 1], [6.922596716, 1.77106367, 1], [8.675418651, -0.242068655, 1],
                   [7.673756466, 3.508563011, 1]]
        coefficients = [-0.406605464, 0.852573316, -1.104746259]

        actual_predictions = list()
        for row in dataset:
            actual_predictions.append(
                predict_with_logistic_regression_classification(row, coefficients)
            )

        expected_predictions = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        for index in range(len(actual_predictions)):
            self.assertEqual(
                round(actual_predictions[index]),
                expected_predictions[index],
            )

    def test_stochastic_gradient_descent_with_logistic_classification_prediction(self):
        dataset = [[2.7810836, 2.550537003, 0], [1.465489372, 2.362125076, 0], [3.396561688, 4.400293529, 0],
                   [1.38807019, 1.850220317, 0], [3.06407232, 3.005305973, 0], [7.627531214, 2.759262235, 1],
                   [5.332441248, 2.088626775, 1], [6.922596716, 1.77106367, 1], [8.675418651, -0.242068655, 1],
                   [7.673756466, 3.508563011, 1]]
        learning_rate = 0.3
        num_of_epochs = 100

        actual_coefficients = stochastic_gradient_descent(dataset, learning_rate, num_of_epochs,
                                                          predict_with_logistic_regression_classification,
                                                          update_coefficients_with_logistic_regression)

        expected_coefficients = [-0.8596443546618897, 1.5223825112460005, -2.218700210565016]
        for index in range(len(actual_coefficients)):
            self.assertAlmostEqual(
                actual_coefficients[index],
                expected_coefficients[index],
                delta=0.000000000000001
            )

    def test_single_perceptron_classification_prediction(self):
        dataset = [[2.7810836, 2.550537003, 0], [1.465489372, 2.362125076, 0], [3.396561688, 4.400293529, 0],
                   [1.38807019, 1.850220317, 0], [3.06407232, 3.005305973, 0], [7.627531214, 2.759262235, 1],
                   [5.332441248, 2.088626775, 1], [6.922596716, 1.77106367, 1], [8.675418651, -0.242068655, 1],
                   [7.673756466, 3.508563011, 1]]
        weights = [-0.1, 0.20653640140000007, -0.23418117710000003]

        actual_predictions = list()
        for row in dataset:
            actual_predictions.append(
                predict_with_single_perceptron_classification(row, weights)
            )

        expected_predictions = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        for index in range(len(actual_predictions)):
            self.assertEqual(
                actual_predictions[index],
                expected_predictions[index],
            )

    def test_stochastic_gradient_descent_with_single_perceptron_classification_prediction(self):
        dataset = [[2.7810836, 2.550537003, 0], [1.465489372, 2.362125076, 0], [3.396561688, 4.400293529, 0],
                   [1.38807019, 1.850220317, 0], [3.06407232, 3.005305973, 0], [7.627531214, 2.759262235, 1],
                   [5.332441248, 2.088626775, 1], [6.922596716, 1.77106367, 1], [8.675418651, -0.242068655, 1],
                   [7.673756466, 3.508563011, 1]]
        learning_rate = 0.1
        num_of_epochs = 5

        actual_weights = stochastic_gradient_descent(dataset, learning_rate, num_of_epochs,
                                                     predict_with_single_perceptron_classification,
                                                     update_weights_with_single_perceptron)

        expected_weights = [-0.1, 0.20653640140000007, -0.23418117710000003]
        for index in range(len(actual_weights)):
            self.assertAlmostEqual(
                actual_weights[index],
                expected_weights[index],
                delta=0.00000000000000001
            )


if __name__ == '__main__':
    unittest.main()
