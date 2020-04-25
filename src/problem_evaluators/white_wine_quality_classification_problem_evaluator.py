from src.algorithm_evaluation import evaluate_regression_algorithm_with_k_fold_cross_validation
from src.algorithms.regression_algorithms import linear_regression_with_stochastic_gradient_descent
from src.data_preprocessing import load_dataset_csv_file, convert_data_to_floats_in_column_range, normalize_dataset, \
    dataset_minmax


def evaluate_candidate_algorithms_for_white_wine_quality_classification_problem(num_cross_validation_folds):
    white_wine_quality_dataset = preprocess_and_normalize_white_wine_quality_dataset()

    evaluate_regression_algorithm_with_k_fold_cross_validation(
        white_wine_quality_dataset,
        linear_regression_with_stochastic_gradient_descent,
        num_cross_validation_folds,
        0.01,
        50
    )


def preprocess_and_normalize_white_wine_quality_dataset():
    dataset = load_dataset_csv_file('datasets/white-wine-quality.data.csv')
    convert_data_to_floats_in_column_range(dataset, range(len(dataset[0])))
    return normalize_dataset(dataset, dataset_minmax(dataset))
