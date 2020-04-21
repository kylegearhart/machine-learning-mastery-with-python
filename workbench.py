from random import seed

from algorithm_evaluation import evaluate_algorithm_with_train_test_split, \
    evaluate_algorithm_with_k_fold_cross_validation, evaluate_regression_algorithm_using_training_dataset
from algorithms.classification_algorithms import zero_rule_algorithm_for_classification
from data_preprocessing import preprocess_and_normalize_pima_indians_diabetes_dataset, \
    preprocess_and_standardize_pima_indians_diabetes_dataset, preprocess_swedish_auto_insurance_dataset
from algorithms.regression_algorithms import simple_linear_regression

seed(1)  # Ensure that results are always the same

train_test_split_percentage = 0.6
num_cross_validation_folds = 5

normalized_pima_dataset = preprocess_and_normalize_pima_indians_diabetes_dataset()
evaluate_algorithm_with_train_test_split(
    normalized_pima_dataset,
    zero_rule_algorithm_for_classification,
    train_test_split_percentage
)
evaluate_algorithm_with_k_fold_cross_validation(
    normalized_pima_dataset,
    zero_rule_algorithm_for_classification,
    num_cross_validation_folds
)

standardized_pima_dataset = preprocess_and_standardize_pima_indians_diabetes_dataset()
evaluate_algorithm_with_train_test_split(
    standardized_pima_dataset,
    zero_rule_algorithm_for_classification,
    train_test_split_percentage
)
evaluate_algorithm_with_k_fold_cross_validation(
    standardized_pima_dataset,
    zero_rule_algorithm_for_classification,
    num_cross_validation_folds
)

swedish_auto_insurance_dataset = preprocess_swedish_auto_insurance_dataset()
evaluate_regression_algorithm_using_training_dataset(
    swedish_auto_insurance_dataset,
    simple_linear_regression
)
