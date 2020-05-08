from src.algorithm_evaluation import evaluate_classification_algorithm_with_k_fold_cross_validation
from src.algorithms.classification_algorithms import zero_rule_algorithm_for_classification, \
    decision_tree
from src.data_preprocessing import load_dataset_csv_file, convert_data_to_floats_in_column_range


def evaluate_candidate_algorithms_for_banknote_authentication_binary_classification_problem(num_cross_validation_folds):
    banknote_authentication_dataset = preprocess_banknote_authentication_dataset()

    evaluate_classification_algorithm_with_k_fold_cross_validation(
        banknote_authentication_dataset,
        zero_rule_algorithm_for_classification,
        num_cross_validation_folds
    )

    max_tree_depth = 5
    min_num_rows_in_node_dataset = 10
    evaluate_classification_algorithm_with_k_fold_cross_validation(
        banknote_authentication_dataset,
        decision_tree,
        num_cross_validation_folds,
        max_tree_depth,
        min_num_rows_in_node_dataset
    )


def preprocess_banknote_authentication_dataset():
    dataset = load_dataset_csv_file('datasets/banknote-authentication.data.csv')
    convert_data_to_floats_in_column_range(dataset, range(0, len(dataset[0])))

    return dataset
