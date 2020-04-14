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


diabetes_csv_filename = 'datasets/pima-indians-diabetes.data.csv'
dataset = load_csv(diabetes_csv_filename)
print('Loaded data file {0} with {1} rows and {2} columns'.format(diabetes_csv_filename, len(dataset), len(dataset[0])))
