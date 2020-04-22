from random import randrange

from src.statistics_utilities import calculate_root_mean_squared_error, calculate_percentage_correct


def evaluate_regression_algorithm_using_training_dataset(training_dataset, algorithm, *args):
    copy_of_training_dataset = training_dataset
    test_dataset_with_predictions = copy_of_training_dataset
    test_dataset = dataset_with_predictions_cleared_out(test_dataset_with_predictions)
    predictions_by_algorithm_on_test_dataset = algorithm(training_dataset, test_dataset, *args)
    print('Predictions made by regression algorithm: \n{0}'.format(predictions_by_algorithm_on_test_dataset))
    prediction_column_index = -1
    correct_predictions_for_test_dataset = [row[prediction_column_index] for row in test_dataset_with_predictions]
    root_mean_squared_error = \
        calculate_regression_prediction_accuracy(predictions_by_algorithm_on_test_dataset,
                                                 correct_predictions_for_test_dataset)
    return root_mean_squared_error


def evaluate_algorithm_with_train_test_split(dataset, algorithm, split_percentage, *args):
    training_dataset, test_dataset_with_predictions = train_test_split(dataset, split_percentage)
    test_dataset = dataset_with_predictions_cleared_out(test_dataset_with_predictions)
    predictions_by_algorithm_on_test_dataset = algorithm(training_dataset, test_dataset, *args)
    prediction_column_index = -1
    correct_predictions_for_test_dataset = [row[prediction_column_index] for row in test_dataset_with_predictions]
    accuracy_of_algorithm = \
        calculate_classification_accuracy(predictions_by_algorithm_on_test_dataset,
                                          correct_predictions_for_test_dataset)
    print('Accuracy for algorithm using {0}% test/training split: '.format(split_percentage * 100))
    print('%.3f%%\n' % accuracy_of_algorithm)
    return accuracy_of_algorithm


def evaluate_algorithm_with_k_fold_cross_validation(dataset, algorithm, num_folds, *args):
    folds = generate_cross_validation_split_data_folds(dataset, num_folds)
    algorithm_accuracy_on_each_fold = list()
    for test_fold in folds:
        training_dataset_folds = list(folds)
        training_dataset_folds.remove(test_fold)
        training_dataset = sum(training_dataset_folds, [])
        test_dataset = dataset_with_predictions_cleared_out(test_fold)
        predictions_by_algorithm_on_test_dataset = algorithm(training_dataset, test_dataset, *args)
        prediction_column_index = -1
        correct_predictions_for_test_dataset = [row[prediction_column_index] for row in test_fold]
        accuracy_of_algorithm_on_current_fold = \
            calculate_classification_accuracy(predictions_by_algorithm_on_test_dataset,
                                              correct_predictions_for_test_dataset)
        algorithm_accuracy_on_each_fold.append(accuracy_of_algorithm_on_current_fold)
    print('Accuracy for algorithm over {0} folds:'.format(num_folds))
    print(' '.join(str('%.3f%%' % accuracy) for accuracy in algorithm_accuracy_on_each_fold))
    print('Mean accuracy: %.3f%%\n' % (sum(algorithm_accuracy_on_each_fold) / num_folds))


def train_test_split(dataset, split_percentage=0.60):
    training_data_rows = list()
    target_num_training_data_rows = split_percentage * len(dataset)
    dataset_copy = list(dataset)
    while len(training_data_rows) < target_num_training_data_rows:
        random_row_index = randrange(len(dataset_copy))
        training_data_rows.append(dataset_copy.pop(random_row_index))
    test_data_rows = dataset_copy
    print('Generated a {0}% training/test data split: \nNum. rows of training data: {1}\nNum. rows of test data: {2}\n'
          .format(int(split_percentage * 100), len(training_data_rows), len(test_data_rows)))
    return training_data_rows, test_data_rows


def generate_cross_validation_split_data_folds(dataset, num_folds=3):
    dataset_folds = list()
    dataset_copy = list(dataset)
    target_num_rows_in_each_fold = int(len(dataset) / num_folds)
    for _ in range(num_folds):
        current_fold = list()
        while len(current_fold) < target_num_rows_in_each_fold:
            random_row_index = randrange(len(dataset_copy))
            current_fold.append(dataset_copy.pop(random_row_index))
        dataset_folds.append(current_fold)
    print('Split {0}-row dataset into {1} folds with {2} rows each:\n({3} rows are being left out of the data folds)\n'
          .format(len(dataset), num_folds, target_num_rows_in_each_fold, int(len(dataset) % num_folds)))
    return dataset_folds


def dataset_with_predictions_cleared_out(dataset):
    dataset_without_predictions = list()
    for row in dataset:
        row_copy = list(row)
        prediction_column_index = -1
        row_copy[prediction_column_index] = None
        dataset_without_predictions.append(row_copy)
    return dataset_without_predictions


def calculate_classification_accuracy(actual_classes, predicted_classes):
    return calculate_percentage_correct(actual_classes, predicted_classes)


def calculate_regression_prediction_accuracy(actual_values, predicted_values):
    return calculate_root_mean_squared_error(actual_values, predicted_values)
