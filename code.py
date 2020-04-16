from csv import reader


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


def convert_entire_column_to_floats(data_rows, column_index):
    for row in data_rows:
        row[column_index] = float(row[column_index].strip())


def convert_data_to_floats_in_column_range(data_rows, column_range):
    for column_index in column_range:
        convert_entire_column_to_floats(data_rows, column_index)
    print('Converted data in columns {0}-{1} to floats'.format(column_range[0], column_range[len(column_range)-1]))
    print_first_five_rows_of_data(data_rows)


def normalize_dataset(dataset, minmax):
    for row in dataset:
        for column_index in range(len(row)):
            min_for_column = minmax[column_index][0]
            max_for_column = minmax[column_index][1]
            row[column_index] = (row[column_index] - min_for_column) / (max_for_column - min_for_column)
    print('Normalized entire dataset using min-maxes of: {0}'.format(minmax))
    print_first_five_rows_of_data(dataset)


def dataset_minmax(dataset):
    minmax = list()
    num_columns = range(len(dataset[0]))
    for column_index in num_columns:
        column_values = [row[column_index] for row in dataset]
        minmax.append([min(column_values), max(column_values)])
    return minmax


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


def preprocess_pima_indians_diabetes_dataset():
    dataset = load_dataset_csv_file('datasets/pima-indians-diabetes.data.csv')
    convert_data_to_floats_in_column_range(dataset,
                                           range(0, len(dataset[0])))
    normalize_dataset(dataset, dataset_minmax(dataset))


def preprocess_iris_flowers_dataset():
    iris_flowers_dataset = load_dataset_csv_file('datasets/iris-species.data.csv')
    convert_string_class_names_to_ints_for_column(iris_flowers_dataset, 4)


preprocess_pima_indians_diabetes_dataset()
preprocess_iris_flowers_dataset()
