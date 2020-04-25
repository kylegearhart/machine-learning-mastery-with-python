from src.algorithm_evaluation import evaluate_regression_algorithm_using_training_dataset
from src.algorithms.regression_algorithms import simple_linear_regression
from src.data_preprocessing import preprocess_swedish_auto_insurance_dataset


def evaluate_candidate_algorithms_for_swedish_auto_insurance_regression_problem(num_cross_validation_folds):
    swedish_auto_insurance_dataset = preprocess_swedish_auto_insurance_dataset()

    evaluate_regression_algorithm_using_training_dataset(
        swedish_auto_insurance_dataset,
        simple_linear_regression
    )
