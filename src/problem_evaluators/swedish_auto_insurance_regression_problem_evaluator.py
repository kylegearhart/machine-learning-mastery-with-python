from src.algorithm_evaluation import evaluate_regression_algorithm_using_training_dataset
from src.algorithms.regression_algorithms import simple_linear_regression
from src.data_preprocessing import load_dataset_csv_file, convert_data_to_floats_in_column_range


def evaluate_candidate_algorithms_for_swedish_auto_insurance_regression_problem(num_cross_validation_folds):
    swedish_auto_insurance_dataset = preprocess_swedish_auto_insurance_dataset()

    evaluate_regression_algorithm_using_training_dataset(
        swedish_auto_insurance_dataset,
        simple_linear_regression
    )


def preprocess_swedish_auto_insurance_dataset():
    dataset = load_dataset_csv_file('datasets/swedish-auto-insurance.data.csv')
    convert_data_to_floats_in_column_range(dataset, range(len(dataset[0])))
    return dataset
