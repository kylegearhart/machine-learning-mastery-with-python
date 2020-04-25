from src.algorithm_evaluation import evaluate_classification_algorithm_with_train_test_split, \
    evaluate_classification_algorithm_with_k_fold_cross_validation
from src.algorithms.classification_algorithms import zero_rule_algorithm_for_classification, \
    logistic_regression_with_stochastic_gradient_descent
from src.data_preprocessing import load_dataset_csv_file, \
    convert_data_to_floats_in_column_range, normalize_dataset, dataset_minmax, column_means_for, column_stdevs_for, \
    standardize_dataset


def evaluate_candidate_algorithms_for_pima_indians_diabetes_binary_classification_problem(num_cross_validation_folds):
    normalized_pima_dataset = preprocess_and_normalize_pima_indians_diabetes_dataset()
    standardized_pima_dataset = preprocess_and_standardize_pima_indians_diabetes_dataset()

    evaluate_classification_algorithm_with_train_test_split(
        normalized_pima_dataset,
        zero_rule_algorithm_for_classification,
        0.6
    )

    evaluate_classification_algorithm_with_k_fold_cross_validation(
        standardized_pima_dataset,
        zero_rule_algorithm_for_classification,
        num_cross_validation_folds
    )

    evaluate_classification_algorithm_with_k_fold_cross_validation(
        normalized_pima_dataset,
        zero_rule_algorithm_for_classification,
        num_cross_validation_folds
    )

    evaluate_classification_algorithm_with_k_fold_cross_validation(
        normalized_pima_dataset,
        logistic_regression_with_stochastic_gradient_descent,
        num_cross_validation_folds,
        .1,
        100
    )


def preprocess_and_normalize_pima_indians_diabetes_dataset():
    dataset = load_dataset_csv_file('datasets/pima-indians-diabetes.data.csv')
    convert_data_to_floats_in_column_range(dataset,
                                           range(0, len(dataset[0])))
    return normalize_dataset(dataset, dataset_minmax(dataset))


def preprocess_and_standardize_pima_indians_diabetes_dataset():
    dataset = load_dataset_csv_file('datasets/pima-indians-diabetes.data.csv')
    convert_data_to_floats_in_column_range(dataset,
                                           range(0, len(dataset[0])))
    column_means = column_means_for(dataset)
    column_stdevs = column_stdevs_for(dataset, column_means)
    return standardize_dataset(dataset, column_means, column_stdevs)
