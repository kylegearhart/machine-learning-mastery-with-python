def stochastic_gradient_descent_coefficients(training_dataset, learning_rate, num_of_epochs, prediction_fn,
                                             coefficient_update_fn):
    correct_value_column_index = -1
    num_of_coefficients = len(training_dataset[0])

    coefficients = [0.0 for _ in range(num_of_coefficients)]
    for epoch in range(num_of_epochs):
        error_squared_sum = 0
        for row in training_dataset:
            prediction = prediction_fn(row, coefficients)
            error = prediction - row[correct_value_column_index]
            error_squared_sum += error ** 2
            coefficient_update_fn(coefficients, row, learning_rate, error)
        print('>epoch=%d, learning_rate=%.3f, error=%.3f' % (epoch, learning_rate, error_squared_sum))
    return coefficients
