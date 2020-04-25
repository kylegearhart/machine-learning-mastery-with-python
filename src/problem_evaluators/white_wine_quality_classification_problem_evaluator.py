from src.algorithm_evaluation import evaluate_regression_algorithm_with_k_fold_cross_validation
from src.algorithms.regression_algorithms import linear_regression_with_stochastic_gradient_descent
from src.data_preprocessing import preprocess_and_normalize_white_wine_quality_dataset


def evaluate_candidate_algorithms_for_white_wine_quality_classification_problem(num_cross_validation_folds):
    white_wine_quality_dataset = preprocess_and_normalize_white_wine_quality_dataset()

    evaluate_regression_algorithm_with_k_fold_cross_validation(
        white_wine_quality_dataset,
        linear_regression_with_stochastic_gradient_descent,
        num_cross_validation_folds,
        0.01,
        50
    )
