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


def convert_all_attributes_to_floats(data_rows):
    num_columns = len(data_rows[0])
    for column_index in range(num_columns):
        convert_entire_column_to_floats(dataset, column_index)


def print_first_five_rows_of_data(data_rows):
    print('First five rows in dataset:')
    for row_index in range(5):
        print(data_rows[0])


diabetes_csv_filename = 'datasets/pima-indians-diabetes.data.csv'
dataset = load_csv(diabetes_csv_filename)
print('Loaded data file {0} with {1} rows and {2} columns'.format(diabetes_csv_filename, len(dataset), len(dataset[0])))
print_first_five_rows_of_data(dataset)

convert_all_attributes_to_floats(dataset)
print_first_five_rows_of_data(dataset)
