from random import seed

from src.algorithm_evaluation import evaluate_classification_algorithm_with_train_test_split, \
    evaluate_classification_algorithm_with_k_fold_cross_validation, \
    evaluate_regression_algorithm_using_training_dataset, evaluate_regression_algorithm_with_k_fold_cross_validation
from src.algorithms.classification_algorithms import zero_rule_algorithm_for_classification
from src.algorithms.regression_algorithms import simple_linear_regression, \
    linear_regression_with_stochastic_gradient_descent
from src.data_preprocessing import preprocess_and_normalize_pima_indians_diabetes_dataset, \
    preprocess_and_standardize_pima_indians_diabetes_dataset, preprocess_swedish_auto_insurance_dataset, \
    preprocess_and_normalize_wine_quality_white_dataset

seed(1)  # Ensure that results are always the same

train_test_split_percentage = 0.6
num_cross_validation_folds = 5

normalized_pima_dataset = preprocess_and_normalize_pima_indians_diabetes_dataset()
evaluate_classification_algorithm_with_train_test_split(
    normalized_pima_dataset,
    zero_rule_algorithm_for_classification,
    train_test_split_percentage
)
evaluate_classification_algorithm_with_k_fold_cross_validation(
    normalized_pima_dataset,
    zero_rule_algorithm_for_classification,
    num_cross_validation_folds
)

standardized_pima_dataset = preprocess_and_standardize_pima_indians_diabetes_dataset()
evaluate_classification_algorithm_with_train_test_split(
    standardized_pima_dataset,
    zero_rule_algorithm_for_classification,
    train_test_split_percentage
)
evaluate_classification_algorithm_with_k_fold_cross_validation(
    standardized_pima_dataset,
    zero_rule_algorithm_for_classification,
    num_cross_validation_folds
)

swedish_auto_insurance_dataset = preprocess_swedish_auto_insurance_dataset()
evaluate_regression_algorithm_using_training_dataset(
    swedish_auto_insurance_dataset,
    simple_linear_regression
)

wine_quality_white_dataset = preprocess_and_normalize_wine_quality_white_dataset()
num_folds = 5
learning_rate = 0.01
num_epochs = 50
evaluate_regression_algorithm_with_k_fold_cross_validation(
    wine_quality_white_dataset,
    linear_regression_with_stochastic_gradient_descent,
    num_folds,
    learning_rate,
    num_epochs
)
