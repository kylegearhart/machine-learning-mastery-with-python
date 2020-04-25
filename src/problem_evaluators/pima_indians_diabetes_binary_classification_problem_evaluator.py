from src.algorithm_evaluation import evaluate_classification_algorithm_with_train_test_split, \
    evaluate_classification_algorithm_with_k_fold_cross_validation
from src.algorithms.classification_algorithms import zero_rule_algorithm_for_classification, \
    logistic_regression_with_stochastic_gradient_descent
from src.data_preprocessing import preprocess_and_normalize_pima_indians_diabetes_dataset, \
    preprocess_and_standardize_pima_indians_diabetes_dataset


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
