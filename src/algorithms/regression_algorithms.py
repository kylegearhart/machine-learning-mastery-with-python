from src.statistics_utilities import mean, covariance, variance


def zero_rule_algorithm_for_regression(training_dataset, test_dataset):
    last_column_index = -1
    all_predicted_values_in_training_dataset = [row[last_column_index] for row in training_dataset]
    mean_of_predicted_values = \
        sum(all_predicted_values_in_training_dataset) / float(len(all_predicted_values_in_training_dataset))
    predictions_on_test_dataset = [mean_of_predicted_values for _ in range(len(test_dataset))]
    print('Zero-Rule prediction on test dataset (regression/mean predicted value): {0}\n'
          .format(predictions_on_test_dataset[0]))
    return predictions_on_test_dataset


def simple_linear_regression(training_dataset, test_dataset):
    y_predictions = list()
    intercept, b1 = linear_coefficients(training_dataset)
    for row in test_dataset:
        x_value = row[0]
        predicted_y = intercept + b1 * x_value
        y_predictions.append(predicted_y)
    return y_predictions


def linear_coefficients(two_column_dataset):
    x_values = [row[0] for row in two_column_dataset]
    y_values = [row[1] for row in two_column_dataset]
    x_mean, y_mean = mean(x_values), mean(y_values)
    b1 = covariance(x_values, x_mean, y_values, y_mean) / variance(x_values, x_mean)
    intercept = y_mean - b1 * x_mean
    return [intercept, b1]
