from math import sqrt


def calculate_percentage_correct(actual_classes, predicted_classes):
    num_of_correct_classifications = 0
    for index in range(len(actual_classes)):
        if actual_classes[index] == predicted_classes[index]:
            num_of_correct_classifications += 1
    return num_of_correct_classifications / float(len(actual_classes)) * 100.0


def covariance(x_values, mean_x, y_values, mean_y):
    result = 0.0
    for index in range(len(x_values)):
        result += ((x_values[index] - mean_x) * (y_values[index] - mean_y))
    return result


def summarize_by_class(dataset):
    class_to_data_rows_dict = separate_dataset_into_classes(dataset)
    class_to_stat_summaries_dict = dict()

    for class_value, data_rows in class_to_data_rows_dict.items():
        class_to_stat_summaries_dict[class_value] = column_stat_summaries(data_rows)

    return class_to_stat_summaries_dict


def column_stat_summaries(dataset):
    stat_summaries = list()
    num_columns_in_dataset = len(dataset[0])
    dataset_columns = zip(*dataset)

    for column_index, column in enumerate(dataset_columns):
        if column_index == (num_columns_in_dataset - 1):
            continue

        stat_summary = (mean(column), standard_deviation(column), len(column))
        stat_summaries.append(stat_summary)

    return stat_summaries


def separate_dataset_into_classes(dataset):
    class_column_index = -1
    class_dict = dict()
    for row_index in range(len(dataset)):
        data_row = dataset[row_index]
        class_value = data_row[class_column_index]

        if class_value not in class_dict:
            class_dict[class_value] = list()

        class_dict[class_value].append(data_row)
    return class_dict


def standard_deviation(numbers):
    return sqrt(variance(numbers, mean(numbers)) / float(len(numbers) - 1))


def variance(values, mean_of_values):
    value_distances_from_mean_squared = [(value - mean_of_values) ** 2 for value in values]
    return sum(value_distances_from_mean_squared)


def mean(values):
    return sum(values) / float(len(values))


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
    print('Confusion matrix:')
    for index1, class1 in enumerate(unique_classes):
        print("%s| %s" % (class1, ' '.join(str(count) for count in confusion_matrix[index1])))
    return unique_classes, confusion_matrix


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
    print('Root mean squared error: %.3f\n' % root_mean_squared_error)
    return root_mean_squared_error
