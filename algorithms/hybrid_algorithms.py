from random import randrange


def random_prediction_algorithm(training_dataset, test_dataset):
    last_column_index = -1
    all_predictions_in_training_dataset = [row[last_column_index] for row in training_dataset]
    set_of_possible_predictions = list(set(all_predictions_in_training_dataset))
    random_predictions_on_test_dataset = list()
    for _ in range(len(test_dataset)):
        random_prediction_index = randrange(len(set_of_possible_predictions))
        random_predictions_on_test_dataset.append(set_of_possible_predictions[random_prediction_index])
    print('Random predictions on test dataset: {0}\n'.format(random_predictions_on_test_dataset))
    return random_predictions_on_test_dataset
