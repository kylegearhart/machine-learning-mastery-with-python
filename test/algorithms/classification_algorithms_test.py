import unittest

from src.algorithms.classification_algorithms import predict_with_logistic_regression_classification


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
                actual_predictions[index],
                expected_predictions[index],
            )


if __name__ == '__main__':
    unittest.main()
