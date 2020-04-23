def stochastic_gradient_descent_coefficients(training_dataset, learning_rate, num_of_epochs, prediction_fn):
    intercept_coefficient_index = 0
    correct_value_column_index = -1
    num_of_coefficients = len(training_dataset[0])

    coefficients = [0.0 for _ in range(num_of_coefficients)]
    for epoch in range(num_of_epochs):
        error_squared_sum = 0
        for row in training_dataset:
            prediction = prediction_fn(row, coefficients)
            error = prediction - row[correct_value_column_index]
            error_squared_sum += error ** 2
            coefficients[intercept_coefficient_index] = \
                coefficients[intercept_coefficient_index] - learning_rate * error
            for index in range(num_of_coefficients - 1):
                coefficients[index + 1] = coefficients[index + 1] - learning_rate * error * row[index]
        print('>epoch=%d, learning_rate=%.3f, error=%.3f' % (epoch, learning_rate, error_squared_sum))
    return coefficients
