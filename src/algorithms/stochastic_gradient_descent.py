def stochastic_gradient_descent(training_dataset, learning_rate, num_of_epochs, prediction_fn, update_fn):
    correct_value_column_index = -1

    coefficients_or_weights = [0.0 for _ in range(len(training_dataset[0]))]
    for epoch in range(num_of_epochs):
        error_squared_sum = 0.0
        for row in training_dataset:
            prediction = prediction_fn(row, coefficients_or_weights)
            error = row[correct_value_column_index] - prediction
            error_squared_sum += error ** 2
            update_fn(coefficients_or_weights, row, learning_rate, prediction, correct_value_column_index)
        print('>epoch=%d, learning_rate=%.3f, error=%.3f' % (epoch, learning_rate, error_squared_sum))

    return coefficients_or_weights
