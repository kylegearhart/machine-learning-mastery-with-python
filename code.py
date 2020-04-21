from csv import reader
from math import sqrt
from random import randrange, seed


def convert_string_class_names_to_ints_for_column(data_rows, column_index):
    all_values_in_column = []
    for row in data_rows:
        all_values_in_column.append(row[column_index])
    set_of_discrete_values_in_column = set(all_values_in_column)

    discrete_value_to_int_dict = dict()
    for index, discrete_value in enumerate(set_of_discrete_values_in_column):
        discrete_value_to_int_dict[discrete_value] = index

    for row in data_rows:
        column_value = row[column_index]
        row[column_index] = discrete_value_to_int_dict[column_value]

    print('Performed string-to-int conversion on column {0}: {1}'.format(column_index,
                                                                         discrete_value_to_int_dict))
    print_first_five_rows_of_data(data_rows)


def convert_entire_column_to_floats(data_rows, column_index):
    for row in data_rows:
        row[column_index] = float(row[column_index].strip())


def convert_data_to_floats_in_column_range(data_rows, column_range):
    for column_index in column_range:
        convert_entire_column_to_floats(data_rows, column_index)
    print('Converted data in columns {0}-{1} to floats'.format(column_range[0], column_range[len(column_range) - 1]))
    print_first_five_rows_of_data(data_rows)


def normalize_dataset(dataset, minmax):
    for row in dataset:
        for column_index in range(len(row)):
            min_for_column = minmax[column_index][0]
            max_for_column = minmax[column_index][1]
            row[column_index] = (row[column_index] - min_for_column) / (max_for_column - min_for_column)
    print('Normalized entire dataset:\nusing min-maxes of {0}'.format(minmax))
    print_first_five_rows_of_data(dataset)
    return dataset


def dataset_minmax(dataset):
    minmax = list()
    num_columns = range(len(dataset[0]))
    for column_index in num_columns:
        column_values = [row[column_index] for row in dataset]
        minmax.append([min(column_values), max(column_values)])
    return minmax


def standardize_dataset(dataset, means, stdevs):
    for row in dataset:
        for column_index in range(len(row)):
            row[column_index] = (row[column_index] - means[column_index]) / stdevs[column_index]
    print('Standardized entire dataset:\nusing means of {0}\nstdevs of {1}'.format(means, stdevs))
    print_first_five_rows_of_data(dataset)
    return dataset


def column_means_for(dataset):
    num_rows_in_dataset = float(len(dataset))
    means_for_all_columns = [0 for _ in column_range_for(dataset)]
    for column_index in column_range_for(dataset):
        column_values = [row[column_index] for row in dataset]
        means_for_all_columns[column_index] = sum(column_values) / num_rows_in_dataset

    return means_for_all_columns


def column_stdevs_for(dataset, column_means):
    stdevs_for_all_columns = [0 for _ in column_range_for(dataset)]
    for column_index in column_range_for(dataset):
        variance = [pow(row[column_index] - column_means[column_index], 2) for row in dataset]
        stdevs_for_all_columns[column_index] = sum(variance)
    stdevs_for_all_columns = [sqrt(column_stdev / (float(len(dataset) - 1))) for column_stdev in stdevs_for_all_columns]

    return stdevs_for_all_columns


def column_range_for(dataset):
    return range(len(dataset[0]))


def print_first_five_rows_of_data(data_rows):
    print('First five rows in dataset:')
    for row_index in range(5):
        print(data_rows[row_index])
    print('')


def load_dataset_csv_file(file_path):
    rows = list()
    with open(file_path, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            rows.append(row)
    data_rows = rows
    print('Loaded data file {0} with {1} rows and {2} columns'.format(file_path, len(data_rows), len(data_rows[0])))
    print_first_five_rows_of_data(data_rows)
    return data_rows


def preprocess_and_normalize_pima_indians_diabetes_dataset():
    dataset = load_dataset_csv_file('datasets/pima-indians-diabetes.data.csv')
    convert_data_to_floats_in_column_range(dataset,
                                           range(0, len(dataset[0])))
    return normalize_dataset(dataset, dataset_minmax(dataset))


def preprocess_and_standardize_pima_indians_diabetes_dataset():
    dataset = load_dataset_csv_file('datasets/pima-indians-diabetes.data.csv')
    convert_data_to_floats_in_column_range(dataset,
                                           range(0, len(dataset[0])))
    column_means = column_means_for(dataset)
    column_stdevs = column_stdevs_for(dataset, column_means)
    return standardize_dataset(dataset, column_means, column_stdevs)


def preprocess_iris_flowers_dataset():
    dataset = load_dataset_csv_file('datasets/iris-species.data.csv')
    convert_string_class_names_to_ints_for_column(dataset, 4)
    return dataset


def preprocess_swedish_auto_insurance_dataset():
    dataset = load_dataset_csv_file('datasets/swedish-auto-insurance.data.csv')
    convert_data_to_floats_in_column_range(dataset, range(len(dataset[0])))
    return dataset


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


def calculate_classification_accuracy(actual_classes, predicted_classes):
    num_of_correct_classifications = 0
    for index in range(len(actual_classes)):
        if actual_classes[index] == predicted_classes[index]:
            num_of_correct_classifications += 1
    classification_accuracy = num_of_correct_classifications / float(len(actual_classes)) * 100.0
    return classification_accuracy


def calculate_regression_prediction_accuracy(actual_values, predicted_values):
    return calculate_root_mean_squared_error(actual_values, predicted_values)


def generate_confusion_matrix(actual_classes, predicted_classes):
    unique_classes = set(actual_classes)
    confusion_matrix = [list() for _ in range(len(unique_classes))]
    for class_index in range(len(unique_classes)):
        confusion_matrix[class_index] = [0 for _ in range(len(unique_classes))]
    class_to_index_lookup_table = dict()
    for class_index, a_class in enumerate(unique_classes):
        class_to_index_lookup_table[a_class] = class_index
    for index in range(len(actual_classes)):
        row_index = class_to_index_lookup_table[actual_classes[index]]
        column_index = class_to_index_lookup_table[predicted_classes[index]]
        confusion_matrix[row_index][column_index] += 1
    pretty_print_confusion_matrix(unique_classes, confusion_matrix)
    return unique_classes, confusion_matrix


def pretty_print_confusion_matrix(unique_classes, confusion_matrix):
    print('Confusion matrix:')
    for row_index, a_class in enumerate(unique_classes):
        print("%s| %s" % (a_class, ' '.join(str(count) for count in confusion_matrix[row_index])))


def calculate_mean_absolute_error(actual_values, predicted_values):
    sum_of_absolute_error_deltas = 0.0
    for index in range(len(actual_values)):
        sum_of_absolute_error_deltas += abs(predicted_values[index] - actual_values[index])
    mean_absolute_error = sum_of_absolute_error_deltas / float(len(actual_values))
    print('Mean absolute error: {0}\n'.format(mean_absolute_error))
    return mean_absolute_error


def calculate_root_mean_squared_error(actual_values, predicted_values):
    sum_of_squared_error_deltas = 0.0
    for index in range(len(actual_values)):
        error_delta = predicted_values[index] - actual_values[index]
        sum_of_squared_error_deltas += error_delta ** 2
    mean_squared_error_delta = sum_of_squared_error_deltas / float(len(actual_values))
    root_mean_squared_error = sqrt(mean_squared_error_delta)
    print('Root mean squared error: {0}\n'.format(root_mean_squared_error))
    return root_mean_squared_error


def random_prediction_algorithm(training_dataset, test_dataset):
    last_column_index = -1
    all_predictions_in_training_dataset = [row[last_column_index] for row in training_dataset]
    set_of_possible_predictions = list(set(all_predictions_in_training_dataset))
    random_predictions_on_test_dataset = list()
    for _ in range(len(test_dataset)):
        random_prediction_index = randrange(len(set_of_possible_predictions))
        random_predictions_on_test_dataset.append(set_of_possible_predictions[random_prediction_index])
    print('Random predictions on test dataset: {0}\n'.format(random_predictions_on_test_dataset))
    return random_predictions_on_test_dataset


def zero_rule_algorithm_for_classification(training_dataset, test_dataset):
    last_column_index = -1
    all_classes_in_training_dataset = [row[last_column_index] for row in training_dataset]
    most_common_class = max(set(all_classes_in_training_dataset), key=all_classes_in_training_dataset.count)
    predictions_on_test_dataset = [most_common_class for _ in range(len(test_dataset))]
    print('Zero-Rule prediction on test dataset (classification/most-common class): {0}\n'
          .format(predictions_on_test_dataset[0]))
    return predictions_on_test_dataset


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


def covariance(x_values, mean_x, y_values, mean_y):
    result = 0.0
    for index in range(len(x_values)):
        result += ((x_values[index] - mean_x) * (y_values[index] - mean_y))
    return result


def variance(values, mean_of_values):
    value_distances_from_mean_squared = [(value - mean_of_values) ** 2 for value in values]
    return sum(value_distances_from_mean_squared)


def mean(values):
    return sum(values) / float(len(values))


def dataset_with_predictions_cleared_out(dataset):
    dataset_without_predictions = list()
    for row in dataset:
        row_copy = list(row)
        prediction_column_index = -1
        row_copy[prediction_column_index] = None
        dataset_without_predictions.append(row_copy)
    return dataset_without_predictions


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
    print('Accuracy for algorithm using {0}% test/training split: '.format(train_test_split_percentage * 100))
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
    print('Mean accuracy: %.3f%%' % (sum(algorithm_accuracy_on_each_fold) / num_folds))


seed(1)  # Ensure that results are always the same

train_test_split_percentage = 0.6
num_cross_validation_folds = 5

normalized_pima_dataset = preprocess_and_normalize_pima_indians_diabetes_dataset()
evaluate_algorithm_with_train_test_split(
    normalized_pima_dataset,
    zero_rule_algorithm_for_classification,
    train_test_split_percentage
)
evaluate_algorithm_with_k_fold_cross_validation(
    normalized_pima_dataset,
    zero_rule_algorithm_for_classification,
    num_cross_validation_folds
)

standardized_pima_dataset = preprocess_and_standardize_pima_indians_diabetes_dataset()
evaluate_algorithm_with_train_test_split(
    standardized_pima_dataset,
    zero_rule_algorithm_for_classification,
    train_test_split_percentage
)
evaluate_algorithm_with_k_fold_cross_validation(
    standardized_pima_dataset,
    zero_rule_algorithm_for_classification,
    num_cross_validation_folds
)

swedish_auto_insurance_dataset = preprocess_swedish_auto_insurance_dataset()
evaluate_regression_algorithm_using_training_dataset(
    swedish_auto_insurance_dataset,
    simple_linear_regression
)
