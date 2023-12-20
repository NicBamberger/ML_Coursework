import argparse
from ML_COURSEWORK.experiments.experiment_base_0 import run_experiment_base
from ML_COURSEWORK.experiments.experiment_cov_split_1 import run_experiment_cov_split_1
from ML_COURSEWORK.experiments.experiment_cov_split_2 import run_experiment_cov_split_2
from ML_COURSEWORK.experiments.experiment_cov_split_3 import run_experiment_cov_split_3
from ML_COURSEWORK.experiments.experiment_cov_split_4 import run_experiment_cov_split_4

def main(datafile, experiment=None):
    if experiment is None or experiment == 'base':
        run_experiment_base(datafile)
    if experiment is None or experiment == 'cov1':
        run_experiment_cov_split_1(datafile)
    if experiment is None or experiment == 'cov2':
        run_experiment_cov_split_2(datafile)
    if experiment is None or experiment == 'cov3':
        run_experiment_cov_split_3(datafile)
    if experiment is None or experiment == 'cov4':
        run_experiment_cov_split_4(datafile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specific ML experiment or all.")
    parser.add_argument("datafile", type=str, help="The path to the data file.")
    parser.add_argument("--experiment", type=str, choices=['base', 'cov1', 'cov2', 'cov3', 'cov4'],
                        help="The specific experiment to run (default: run all).")

    args = parser.parse_args()
    main(args.datafile, args.experiment)