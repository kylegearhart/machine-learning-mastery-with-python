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