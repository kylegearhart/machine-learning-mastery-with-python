from src.algorithm_evaluation import evaluate_classification_algorithm_with_k_fold_cross_validation
from src.algorithms.classification_algorithms import zero_rule_algorithm_for_classification, \
    single_perceptron_with_stochastic_gradient_descent
from src.data_preprocessing import load_dataset_csv_file, convert_data_to_floats_in_column_range, \
    convert_string_class_names_to_ints_for_column


def evaluate_candidate_algorithms_for_mine_or_rock_sonar_binary_classification_problem(num_cross_validation_folds):
    mine_or_rock_sonar_dataset = preprocess_mine_or_rock_sonar_dataset()

    evaluate_classification_algorithm_with_k_fold_cross_validation(
        mine_or_rock_sonar_dataset,
        zero_rule_algorithm_for_classification,
        num_cross_validation_folds
    )

    evaluate_classification_algorithm_with_k_fold_cross_validation(
        mine_or_rock_sonar_dataset,
        single_perceptron_with_stochastic_gradient_descent,
        num_cross_validation_folds,
        0.01,
        500
    )


def preprocess_mine_or_rock_sonar_dataset():
    dataset = load_dataset_csv_file('datasets/mine-or-rock-sonar.data.csv')
    convert_data_to_floats_in_column_range(dataset,
                                           range(0, len(dataset[0]) - 1))
    convert_string_class_names_to_ints_for_column(dataset, -1)

    return dataset
