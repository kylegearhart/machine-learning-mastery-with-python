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


dataset = load_dataset_csv_file('datasets/pima-indians-diabetes.data.csv')
print_first_five_rows_of_data(dataset)

convert_data_to_floats_in_column_range(dataset, range(0, len(dataset[0])))
print_first_five_rows_of_data(dataset)
