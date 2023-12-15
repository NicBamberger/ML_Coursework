import time
import pandas as pd
import numpy as np
import scipy.stats
from sklearn import ensemble
import matplotlib.pyplot as plt
import os

# functions we have created with the class "Gx"
from functions.pipeline import *

start = time.time()
SEED = 123
np.random.seed(SEED)


def load_data():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(script_dir, "data", "data.csv")
    df = pd.read_csv(path)
    matrix, target = Gx.processing(df)

    return matrix, target


def feature_sets(matrix, n_splits=4):
    # create a variable number of features by splitting the data into time-bands and then calculate statistics on each time band
    n_splits = 4
    feature_sets = Gx.create_feature_sets(matrix, n_splits)
    feature_set_names = ["base"] + [f"covs_split_{i}" for i in range(1, n_splits + 1)]
    feature_set_dict = dict(zip(feature_set_names, feature_sets))

    return feature_set_dict


def apply_grid_search(feature_sets, target):

    # apply grid search to all our sets of features
    feature_valid_errors = []
    optimals = []
    for i, features in enumerate(feature_sets.values()):
        print(f"\n\nSet {i+1}:\n")
        # contains the optimal hyperparameters
        # and the mean and stdev (of the errors from each fold)
        all_errors, optimal = Gx.grid_search(features, target)  

        optimals.append(optimal) 
        feature_valid_errors.append(optimal.mean_error)
    
    return optimals, feature_valid_errors


def choose_best_model(feature_set_dict, optimals, feature_valid_errors):

    hyperparameter_choices = dict(zip(feature_set_dict.keys(), optimals))
    valid_errors = dict(zip(feature_set_dict.keys(), feature_valid_errors))
    lowest_error_idx = pd.Series(valid_errors).argmin()
    optimal_feature_set_name = list(valid_errors)[lowest_error_idx]

    # print baseline error
    print(f"Baseline model error: {round(hyperparameter_choices['base'].mean_error, 3)}")

    # calculating confidence interval
    z = scipy.stats.norm.ppf(.95)
    optimal_hps = hyperparameter_choices[optimal_feature_set_name]

    mu = optimal_hps.mean_error
    sigma = optimal_hps.stdev_error 
    lower_conf = mu - z * sigma
    lower_conf = round(max(0, lower_conf), 3) # can't have negative error!
    upper_conf = round(mu + z * sigma, 3)
    conf_interval = (lower_conf, upper_conf)
    
    return optimal_feature_set_name, mu, conf_interval


def test_results(
    features, 
    target, 
    optimal_hps
    ):

    X_train, y_train, X_test, y_test = Gx.train_test_split(features, target)

    model = ensemble.GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS, 
        max_depth=int(optimal_hps.loc["max_depth"]), 
        min_impurity_decrease=float(optimal_hps.loc["min_impurity_decrease"]), 
        loss="log_loss"
        )

    error_test, y_test_probs = Gx.run_model(X_train, y_train, X_test, y_test, model)

    return error_test, y_test_probs


def main():
    
    matrix, target = load_data()
    feature_set_dict = feature_sets(matrix, n_splits=4)
    optimals, feature_valid_errors = apply_grid_search(feature_set_dict, target)
    optimal_feature_set_name, mu, conf_interval = choose_best_model(feature_set_dict, optimals, feature_valid_errors)
    
    print(f"Best feature set: {optimal_feature_set_name}")
    print(f"Mean error and 95% confidence interval of error: {round(mu, 3)}, {conf_interval}")
    
    optimal_dict = dict(zip(feature_set_dict.keys(), optimals))
    optimal_hps = optimal_dict[optimal_feature_set_name] 
    features = feature_set_dict[optimal_feature_set_name]

    error_test, y_test_probs = test_results(features, target, optimal_hps)

    print(f"Test error (log loss): {round(error_test, 3)}")

    pd.DataFrame(y_test_probs).plot()
    plt.show()

main()
