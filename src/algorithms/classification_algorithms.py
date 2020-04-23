from math import exp


def zero_rule_algorithm_for_classification(training_dataset, test_dataset):
    last_column_index = -1
    all_classes_in_training_dataset = [row[last_column_index] for row in training_dataset]
    most_common_class = max(set(all_classes_in_training_dataset), key=all_classes_in_training_dataset.count)
    predictions_on_test_dataset = [most_common_class for _ in range(len(test_dataset))]
    print('Zero-Rule prediction on test dataset (classification/most-common class): {0}\n'
          .format(predictions_on_test_dataset[0]))
    return predictions_on_test_dataset


def predict_with_logistic_regression_classification(row, coefficients):
    intercept = coefficients[0]

    prediction_as_float = intercept
    for value_index in range(len(row) - 1):
        prediction_as_float += coefficients[value_index + 1] * row[value_index]
    prediction_as_float = 1.0 / (1.0 + exp(-prediction_as_float))

    prediction_as_class = round(prediction_as_float)
    return prediction_as_class
