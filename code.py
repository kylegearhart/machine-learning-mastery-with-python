from csv import reader


def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    return list(lines)


diabetes_csv_filename = 'datasets/pima-indians-diabetes.data.csv'
dataset = load_csv(diabetes_csv_filename)
print('Loaded data file {0} with {1} rows and {2} columns'.format(diabetes_csv_filename, len(dataset), len(dataset[0])))
