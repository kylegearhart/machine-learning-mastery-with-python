from random import seed

from src.algorithm_evaluation import evaluate_classification_algorithm_with_train_test_split, \
    evaluate_classification_algorithm_with_k_fold_cross_validation, \
    evaluate_regression_algorithm_using_training_dataset, evaluate_regression_algorithm_with_k_fold_cross_validation
from src.algorithms.classification_algorithms import zero_rule_algorithm_for_classification, \
    logistic_regression_with_stochastic_gradient_descent
from src.algorithms.regression_algorithms import simple_linear_regression, \
    linear_regression_with_stochastic_gradient_descent
from src.data_preprocessing import preprocess_and_normalize_pima_indians_diabetes_dataset, \
    preprocess_and_standardize_pima_indians_diabetes_dataset, preprocess_swedish_auto_insurance_dataset, \
    preprocess_and_normalize_wine_quality_white_dataset

seed(1)  # Ensure that results are always the same

num_cross_validation_folds = 5


def evaluate_candidate_algorithms_for_pima_indians_diabetes_binary_classification_problem():
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


def evaluate_candidate_algorithms_for_swedish_auto_insurance_regression_problem():
    swedish_auto_insurance_dataset = preprocess_swedish_auto_insurance_dataset()
    evaluate_regression_algorithm_using_training_dataset(
        swedish_auto_insurance_dataset,
        simple_linear_regression
    )


def evaluate_candidate_algorithms_for_wine_quality_white_classification_problem():
    wine_quality_white_dataset = preprocess_and_normalize_wine_quality_white_dataset()
    evaluate_regression_algorithm_with_k_fold_cross_validation(
        wine_quality_white_dataset,
        linear_regression_with_stochastic_gradient_descent,
        num_cross_validation_folds,
        0.01,
        50
    )


evaluate_candidate_algorithms_for_pima_indians_diabetes_binary_classification_problem()
evaluate_candidate_algorithms_for_wine_quality_white_classification_problem()
evaluate_candidate_algorithms_for_swedish_auto_insurance_regression_problem()
