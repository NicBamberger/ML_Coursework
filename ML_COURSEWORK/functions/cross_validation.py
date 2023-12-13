import numpy as np


def cross_validation_splits(train_validation_data, targets, number_of_folds):
    
    # Convert input data to numpy array if it's not already
    train_validation_data = np.array(train_validation_data)
    targets = np.array(targets)

    # Check if input_data and targets have the same number of rows
    if train_validation_data.shape[0] != targets.shape[0]:
        print("Input Data and Targets do not have the same number of entries.")
        print(f"input_data.shape = {train_validation_data.shape}")

    # Randomly assign each data point to a fold
    fold_assignments = np.random.randint(0, number_of_folds, size=targets.size)
    print(fold_assignments)

    # Saving the different splits in a list
    folds = []

    for f in range(number_of_folds):
        train_filter = (fold_assignments != f)
        valid_filter = ~train_filter

        train_inputs = train_validation_data[train_filter, :]
        train_targets = targets[train_filter]
        valid_inputs = train_validation_data[valid_filter, :]
        valid_targets = targets[valid_filter]

        fold = {
            "train_inputs": train_inputs,
            "train_targets": train_targets,
            "valid_inputs": valid_inputs,
            "valid_targets": valid_targets
        }
        folds.append(fold)

        print(f"For fold {f}")
        print(f"\ttrain_inputs.shape = {train_inputs.shape}")
        print(f"\ttrain_targets.shape = {train_targets.shape}")
        print(f"\tvalid_inputs.shape = {valid_inputs.shape}")
        print(f"\tvalid_targets.shape = {valid_targets.shape}")

    return folds