from csv import reader


def load_csv(filename):
    data_rows = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            data_rows.append(row)
    return data_rows


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


def print_first_five_rows_of_data(data_rows):
    print('First five rows in dataset:')
    for row_index in range(5):
        print(data_rows[row_index])


def load_dataset_csv_file(file_path):
    data_rows = load_csv(file_path)
    print('Loaded data file {0} with {1} rows and {2} columns'.format(file_path, len(data_rows), len(data_rows[0])))
    return data_rows


pima_indians_diabetes_dataset = load_dataset_csv_file('datasets/pima-indians-diabetes.data.csv')
print_first_five_rows_of_data(pima_indians_diabetes_dataset)

convert_data_to_floats_in_column_range(pima_indians_diabetes_dataset, range(0, len(pima_indians_diabetes_dataset[0])))
print_first_five_rows_of_data(pima_indians_diabetes_dataset)

iris_flowers_dataset = load_dataset_csv_file('datasets/iris-species.data.csv')
print_first_five_rows_of_data(iris_flowers_dataset)

convert_string_class_names_to_ints_for_column(iris_flowers_dataset, 4)
print_first_five_rows_of_data(iris_flowers_dataset)
