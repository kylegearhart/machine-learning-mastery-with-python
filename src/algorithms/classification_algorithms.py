from math import exp

from src.algorithms.stochastic_gradient_descent import stochastic_gradient_descent


def zero_rule_algorithm_for_classification(training_dataset, test_dataset):
    last_column_index = -1
    all_classes_in_training_dataset = [row[last_column_index] for row in training_dataset]
    most_common_class = max(set(all_classes_in_training_dataset), key=all_classes_in_training_dataset.count)
    predictions_on_test_dataset = [most_common_class for _ in range(len(test_dataset))]
    print('Zero-Rule prediction on test dataset (classification/most-common class): {0}\n'
          .format(predictions_on_test_dataset[0]))
    return predictions_on_test_dataset


def logistic_regression_with_stochastic_gradient_descent(training_dataset, test_dataset, learning_rate, num_of_epochs):
    predicted_classes = list()
    coefficients = stochastic_gradient_descent(training_dataset, learning_rate, num_of_epochs,
                                               predict_with_logistic_regression_classification,
                                               update_coefficients_with_logistic_regression)

    for row in test_dataset:
        prediction = predict_with_logistic_regression_classification(row, coefficients)
        predicted_class = round(prediction)
        predicted_classes.append(predicted_class)
    return predicted_classes


def predict_with_logistic_regression_classification(row, coefficients):
    intercept = coefficients[0]

    prediction_as_float = intercept
    for value_index in range(len(row) - 1):
        prediction_as_float += coefficients[value_index + 1] * row[value_index]

    prediction_as_float = 1.0 / (1.0 + exp(-prediction_as_float))
    return prediction_as_float


def update_coefficients_with_logistic_regression(coefficients, data_row, learning_rate, predicted_value,
                                                 correct_value_column_index):
    error = data_row[correct_value_column_index] - predicted_value
    intercept_coefficient_index = 0
    coefficients[intercept_coefficient_index] = \
        coefficients[intercept_coefficient_index] + learning_rate * error * predicted_value * (1.0 - predicted_value)
    for index in range(len(data_row) - 1):
        coefficients[index + 1] = coefficients[index + 1] + learning_rate * error * predicted_value * \
                                  (1.0 - predicted_value) * data_row[index]


def single_perceptron_with_stochastic_gradient_descent(training_dataset, test_dataset, learning_rate,
                                                       num_of_epochs):
    predicted_classes = list()
    weights = stochastic_gradient_descent(training_dataset, learning_rate, num_of_epochs,
                                          predict_with_single_perceptron_classification,
                                          update_weights_with_single_perceptron)

    for row in test_dataset:
        predicted_class = predict_with_single_perceptron_classification(row, weights)
        predicted_classes.append(predicted_class)
    return predicted_classes


def predict_with_single_perceptron_classification(row, weights):
    activation = weights[0]
    for data_value_index in range(len(row) - 1):
        activation += weights[data_value_index + 1] * row[data_value_index]
    return 1.0 if activation >= 0.0 else 0.0


def update_weights_with_single_perceptron(weights, data_row, learning_rate, predicted_value,
                                          correct_value_column_index):
    error = data_row[correct_value_column_index] - predicted_value
    bias_weight_index = 0
    weights[bias_weight_index] = weights[bias_weight_index] + learning_rate * error
    for index in range(len(data_row) - 1):
        weights[index + 1] = weights[index + 1] + learning_rate * error * data_row[index]

    return weights
