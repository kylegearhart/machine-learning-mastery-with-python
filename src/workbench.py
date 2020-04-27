from random import seed

from src.problem_evaluators.mine_or_rock_binary_classification_problem_evaluator import \
    evaluate_candidate_algorithms_for_mine_or_rock_sonar_binary_classification_problem
from src.problem_evaluators.pima_indians_diabetes_binary_classification_problem_evaluator import \
    evaluate_candidate_algorithms_for_pima_indians_diabetes_binary_classification_problem
from src.problem_evaluators.swedish_auto_insurance_regression_problem_evaluator import \
    evaluate_candidate_algorithms_for_swedish_auto_insurance_regression_problem
from src.problem_evaluators.white_wine_quality_classification_problem_evaluator import \
    evaluate_candidate_algorithms_for_white_wine_quality_classification_problem

seed(1)  # Ensure that results are always the same

num_cross_validation_folds = 5

evaluate_candidate_algorithms_for_pima_indians_diabetes_binary_classification_problem(num_cross_validation_folds)
evaluate_candidate_algorithms_for_white_wine_quality_classification_problem(num_cross_validation_folds)
evaluate_candidate_algorithms_for_swedish_auto_insurance_regression_problem(num_cross_validation_folds)
evaluate_candidate_algorithms_for_mine_or_rock_sonar_binary_classification_problem(num_cross_validation_folds)
