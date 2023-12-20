from .functions.pipeline import Gx

def run_experiment_cov_split_2(datafile):
    matrix, target = Gx.load_data(datafile)
    feature_set_dict = Gx.feature_sets(matrix, n_splits=4)
    covs_2_features = feature_set_dict['covs_split_2']
    
    # Call apply_grid_search with the base features
    print()
    print("Experiment 3 / 5")
    print("(For Covs_2_Split Experiment (Windowing) Model)")
    print("Grid Search and Cross Validation")
    print()

    all_errors, optimal, feature_valid_error = Gx.apply_grid_search(covs_2_features, target)
    conf_interval = Gx.calculate_confidence_interval(optimal)
    print()
    print("(Optimal Set of Hyperparameters)")
    print(optimal)
    print()
    print("(Covs_2_Split Experiment Mean Validation Error (log-loss)):")
    print(feature_valid_error)
    print()
    print("95% Confidence Interval for Mean Validation Error")
    print(conf_interval)
    print()
    
    error_test, y_test_probs = Gx.test_results(covs_2_features, target, optimal)
    
    print("Testing Model on unseen Data")
    print("(Covs_2_Split Experiment Test Error (log-loss))")
    print(error_test)
    print()

if __name__ == "__main__":
    import sys
    datafile = sys.argv[1]
    run_experiment_cov_split_2(datafile)
