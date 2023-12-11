import numpy as np


def test_validation_split(input_data, target_data, test_percentage):
    
    # Ensure format is correct
    input_data = np.array(input_data)
    target_data = np.array(target_data)

    # Calculate the number of test samples from full dataset
    num_test = int(len(input_data) * (test_percentage / 100))

    # Shuffle the data
    indices = np.arange(len(input_data))
    np.random.shuffle(indices)
    input_data_shuffled = input_data[indices]
    target_data_shuffled = target_data[indices]

    # Split the data
    test_inputs = input_data_shuffled[:num_test]
    val_inputs = input_data_shuffled[num_test:]

    test_targets = target_data_shuffled[:num_test]
    val_targets = target_data_shuffled[num_test:]

    return test_inputs, val_inputs, test_targets, val_targets