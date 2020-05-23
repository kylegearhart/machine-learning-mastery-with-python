from src.algorithm_evaluation import evaluate_classification_algorithm_with_k_fold_cross_validation
from src.algorithms.classification_algorithms import zero_rule_algorithm_for_classification, \
    naive_bayes
from src.data_preprocessing import load_dataset_csv_file, convert_data_to_floats_in_column_range, \
    convert_string_class_names_to_ints_for_column


def evaluate_candidate_algorithms_for_iris_flower_species_classification_problem(num_cross_validation_folds):
    iris_flower_species_dataset = preprocess_iris_flower_species_dataset()

    evaluate_classification_algorithm_with_k_fold_cross_validation(
        iris_flower_species_dataset,
        zero_rule_algorithm_for_classification,
        num_cross_validation_folds
    )

    evaluate_classification_algorithm_with_k_fold_cross_validation(
        iris_flower_species_dataset,
        naive_bayes,
        num_cross_validation_folds
    )


def preprocess_iris_flower_species_dataset():
    dataset = load_dataset_csv_file('datasets/iris-species.data.csv')
    convert_data_to_floats_in_column_range(dataset, range(0, len(dataset[0]) - 1))
    convert_string_class_names_to_ints_for_column(dataset, -1)

    return dataset
