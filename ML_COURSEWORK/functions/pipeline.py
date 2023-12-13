import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

import os

N_SUBJECTS = 10
N_CONDITIONS = 3
N_REPS = 10
N_SENSORS = 6
N_TRIALS = N_SUBJECTS * N_CONDITIONS * N_REPS

TRAIN_SUBJECTS = 7
TRAIN_TRIALS = N_TRIALS * TRAIN_SUBJECTS / N_SUBJECTS

# functions that our group have created

class Gx:

    def processing(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        
        trials = ["subject", "condition", "replication"]
        index_columns = trials + ["time"]
        df = df.set_index(index_columns)

        matrix = pd.DataFrame()
        joint_map = {1: "ankle", 2: "knee", 3: "hip"}
        leg_map = {1: "left", 2: "right"}

        for leg_key, leg_val in leg_map.items():
            for joint_key, joint_val in joint_map.items():
                matrix[f"{leg_val}_{joint_val}"] = df[(df.leg == leg_key) & (df.joint == joint_key)].angle

        rs = matrix.reset_index()
        series = pd.DataFrame((rs.subject - 1) * N_REPS * N_CONDITIONS + (rs.condition - 1) * N_REPS + rs.replication)
        matrix["trial"] = series.set_index(matrix.index)

        target = pd.Series(range(N_TRIALS), index=range(1, N_TRIALS + 1))
        target = 1 + ((target // N_REPS) % N_CONDITIONS)
        target.name = "condition"

        matrix = matrix.reset_index().drop(trials, axis=1).set_index(["trial", "time"])

        return matrix, target
    
    def get_time_splits(
        matrix: pd.DataFrame, 
        n: int
        ):
    
        time = matrix.index.get_level_values(1)
        n_time_points = 101 // n
        time_splits = np.array([None for i in range(n)])

        for i in range(n):
            lower = (i * n_time_points)
            upper = ((i+1) * n_time_points)

            if i == (n - 1):
                upper = 101

            time_splits[i] = matrix[(time >= lower) & (time < upper)]
        
        return time_splits
    
    def cross_validation_splits(
        CV_X: pd.DataFrame, 
        CV_y: pd.Series, 
        n_folds: int
        ) -> list:
        
        # Convert input data to numpy array, but keep the index
        train_validation_index = CV_X.index
        CV_X = np.array(CV_X)
        CV_y = np.array(CV_y)
    
        # Check if input_data and targets have the same number of rows
        if CV_X.shape[0] != CV_y.shape[0]:
            print("Input Data and Targets do not have the same number of entries.")
            print(f"input_data.shape = {CV_X.shape}")

        #  assign each data point to a fold
        # fold_assignments = np.random.randint(0, n_folds, size=y_CV.size)
        fold_assignments = (train_validation_index.values - 1) // (N_REPS * N_CONDITIONS)
        # print(fold_assignments)

        # Saving the different splits in a list
        folds = []

        for f in range(n_folds):
            train_filter = (fold_assignments != f)
            valid_filter = ~train_filter

            CV_X_train = CV_X[train_filter, :]
            CV_y_train = CV_y[train_filter]
            CV_X_valid = CV_X[valid_filter, :]
            CV_y_valid = CV_y[valid_filter]

            fold = {
                "train_inputs": CV_X_train,
                "train_targets": CV_y_train,
                "valid_inputs": CV_X_valid,
                "valid_targets": CV_y_valid
            }
            folds.append(fold)

            # test one fold
            if f == 0:
                print(f"For fold {f}")
                print(f"\ttrain_inputs.shape = {CV_X_train.shape}")
                print(f"\ttrain_targets.shape = {CV_y_train.shape}")
                print(f"\tvalid_inputs.shape = {CV_X_valid.shape}")
                print(f"\tvalid_targets.shape = {CV_y_valid.shape}")

        return folds

    def train_test_split(
        features: pd.DataFrame, 
        target: pd.Series
        ):

        X = features.copy()
        y = target.copy()

        X_train = X.loc[:TRAIN_TRIALS]
        y_train = y.loc[:TRAIN_TRIALS]
        X_test = X.loc[TRAIN_TRIALS+1:]
        y_test = y.loc[TRAIN_TRIALS +1:]

        return X_train, y_train, X_test, y_test

    def run_model(
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame, 
        y_test: pd.Series, 
        model
        ):

        model.fit(X_train, y_train)
        y_probs = model.predict_proba(X_test)
        error = log_loss(y_test, y_probs) # wants clear divisions between classes - would work
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc}")

        return error, y_probs
