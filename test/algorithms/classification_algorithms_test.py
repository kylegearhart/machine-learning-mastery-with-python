import unittest

from src.algorithms.classification_algorithms import predict_with_logistic_regression_classification, \
    update_coefficients_with_logistic_regression, predict_with_single_perceptron_classification, \
    update_weights_with_single_perceptron, calculate_gini_index, split_dataset_on, \
    create_tree_node_with_optimal_split, terminal_node_representing_dataset, build_decision_tree, \
    predict_with_decision_tree, calculate_class_probabilities_using_naive_bayes
from src.algorithms.stochastic_gradient_descent import stochastic_gradient_descent
from src.statistics_utilities import class_stat_summaries


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

    def test_decision_tree_prediction_with_decision_stump(self):
        dataset = [[2.771244718, 1.784783929, 0], [1.728571309, 1.169761413, 0], [3.678319846, 2.81281357, 0],
                   [3.961043357, 2.61995032, 0], [2.999208922, 2.209014212, 0], [7.497545867, 3.162953546, 1],
                   [9.00220326, 3.339047188, 1], [7.444542326, 0.476683375, 1], [10.12493903, 3.234550982, 1],
                   [6.642287351, 3.319983761, 1]]

        decision_stump = {'property_index_to_split_on': 0, 'threshold_value': 6.642287351,
                          'left_subtree': 0, 'right_subtree': 1}

        class_column_index = -1
        for row in dataset:
            predicted_class = predict_with_decision_tree(decision_stump, row)
            correct_prediction = row[class_column_index]
            self.assertEqual(predicted_class, correct_prediction)

    def test_build_decision_tree_root_with_two_terminal_nodes(self):
        dataset = [[2.771244718, 1.784783929, 0], [1.728571309, 1.169761413, 0], [3.678319846, 2.81281357, 0],
                   [3.961043357, 2.61995032, 0], [2.999208922, 2.209014212, 0], [7.497545867, 3.162953546, 1],
                   [9.00220326, 3.339047188, 1], [7.444542326, 0.476683375, 1], [10.12493903, 3.234550982, 1],
                   [6.642287351, 3.319983761, 1]]

        root_node_of_decision_tree = build_decision_tree(dataset, 1, 1)

        self.assertEqual(root_node_of_decision_tree['property_index_to_split_on'], 0)
        self.assertEqual(root_node_of_decision_tree['threshold_value'], 6.642287351)
        self.assertEqual(root_node_of_decision_tree['left_subtree'], 0)
        self.assertEqual(root_node_of_decision_tree['right_subtree'], 1)

    def test_split_dataset(self):
        property_index_to_split_on = 0
        property_threshold_value_to_split_on = 5
        dataset = [[2], [4], [5], [6], [9], [10]]

        left_subtree, right_subtree = split_dataset_on(property_index_to_split_on, property_threshold_value_to_split_on,
                                                       dataset)

        self.assertEqual(left_subtree, [[2], [4]])
        self.assertEqual(right_subtree, [[5], [6], [9], [10]])

    def test_determine_optimal_split_attribute_and_threshold(self):
        dataset = [[2.771244718, 1.784783929, 0], [1.728571309, 1.169761413, 0], [3.678319846, 2.81281357, 0],
                   [3.961043357, 2.61995032, 0], [2.999208922, 2.209014212, 0], [7.497545867, 3.162953546, 1],
                   [9.00220326, 3.339047188, 1], [7.444542326, 0.476683375, 1], [10.12493903, 3.234550982, 1],
                   [6.642287351, 3.319983761, 1]]

        actual_split_data = create_tree_node_with_optimal_split(dataset)

        self.assertEqual(actual_split_data['property_index_to_split_on'], 0)
        self.assertEqual(actual_split_data['threshold_value'], 6.642287351)
        self.assertEqual(
            actual_split_data['subtree_datasets'][0],
            [[2.771244718, 1.784783929, 0], [1.728571309, 1.169761413, 0], [3.678319846, 2.81281357, 0],
             [3.961043357, 2.61995032, 0], [2.999208922, 2.209014212, 0]]
        )
        self.assertEqual(
            actual_split_data['subtree_datasets'][1],
            [[7.497545867, 3.162953546, 1], [9.00220326, 3.339047188, 1], [7.444542326, 0.476683375, 1],
             [10.12493903, 3.234550982, 1], [6.642287351, 3.319983761, 1]]
        )

    def test_terminal_node_class_value(self):
        terminal_node_dataset = [[1.728571309, 1.169761413, 0], [3.678319846, 2.81281357, 0],
                                 [3.961043357, 2.61995032, 0], [2.999208922, 2.209014212, 0],
                                 [7.497545867, 3.162953546, 1], [9.00220326, 3.339047188, 1],
                                 [7.444542326, 0.476683375, 1], [10.12493903, 3.234550982, 1],
                                 [6.642287351, 3.319983761, 1]]

        terminal_node_class = terminal_node_representing_dataset(terminal_node_dataset)

        self.assertEqual(terminal_node_class, 1)

    def test_gini_index_calculation_for_worst_case_class_groupings(self):
        class_list = [0, 1]
        perfect_class_groupings = [[[1, 1], [1, 0]], [[1, 1], [1, 0]]]

        self.assertEqual(calculate_gini_index(perfect_class_groupings, class_list), 0.5)

    def test_gini_index_calculation_for_best_case_class_groupings(self):
        class_list = [0, 1]
        perfect_class_groupings = [[[1, 0], [0, 0]], [[1, 1], [0, 1]]]

        self.assertEqual(calculate_gini_index(perfect_class_groupings, class_list), 0.0)

    def test_naive_bayes_class_probabilities_for_data_row(self):
        dataset = [[3.393533211, 2.331273381, 0], [3.110073483, 1.781539638, 0], [1.343808831, 3.368360954, 0],
                   [3.582294042, 4.67917911, 0], [2.280362439, 2.866990263, 0], [7.423436942, 4.696522875, 1],
                   [5.745051997, 3.533989803, 1], [9.172168622, 2.511101045, 1], [7.792783481, 3.424088941, 1],
                   [7.939820817, 0.791637231, 1]]

        class_summaries = class_stat_summaries(dataset)
        self.assertEqual(
            {0: 0.05032427673372075, 1: 0.00011557718379945765},
            calculate_class_probabilities_using_naive_bayes(class_summaries, dataset[0])
        )


if __name__ == '__main__':
    unittest.main()
