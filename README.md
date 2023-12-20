# Multivariate Gait Data Classification

## Overview

This repository contains the code for a machine learning classification project as part of the Foundations of Machine Learning course at UCL. This project involves using multivariate Gait data from the **UC Irvine Machine Learning Repository** (link provided below) and aims to develop a classification algorithm using a **Gradient Boosting Classifier** to identify lowerbody bracing conditions. The project structure allows for running a series of experiments individually or sequentially through a command-line interface. More details regarding using this interface will be provided below.

## Project Structure

- **Data**: This directory holds the Multivariate Gait Dataset obtained from the UC Irvine Machine Learning Repository (https://archive.ics.uci.edu/dataset/760/multivariate+gait+data).

- **File System**: 
'''
ML_COURSEWORK/
│
├── Content/
│   ├── data/
│   │   └── data.csv
│   │
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── experiment_base_0.py
│   │   ├── experiment_cov_split_1.py
│   │   ├── experiment_cov_split_2.py
│   │   ├── experiment_cov_split_3.py
│   │   └── experiment_cov_split_4.py
│   │
│   └── functions/
│       ├── __init__.py
│       └── pipeline.py
│
├── venv/
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
'''

1. **`main.py`**
   - The main runner script that uses command-line arguments to execute specific experiments.

2. **`data`**
   - This directory contains the dataset data.csv used for the machine learning experiments.

3. **`experiments/`**
   - Contains individual Python scripts for each experiment (experiment_base_0.py, experiment_cov_split_1.py, etc.).

4. **`functions/`**
   - Includes pipeline.py which contains the Gx class with all methods required for data processing, feature extraction, model training, and evaluation.

5. **`functions/`**
   - Lists all the necessary Python dependencies for the project.


## Project Structure

To set up the project environment, follow these steps:

1. Ensure that Python 3.8 or above is installed on your system.
2. Clone the repository to your local machine.
3. Navigate to the project directory and create a virtual environment:
   - ***python -m venv venv***
4. Activate the Virtual Environment:
   On windows:
   - ***venv\Scripts\activate***
     On macOS or Linux:
   - ***source venv/bin/activate***
6. Install the required dependencies:
   - ***pip install -r requirements.txt***

## Running Experiments

To run experiments, use the following command structure from the ML_COURSEWORK directory:

***python main.py ML_COURSEWORK/data/data.csv --experiment <EXPERIMENT_NAME>***

Where <DATA_FILE> is the path to the dataset and <EXPERIMENT_NAME> is one of the following:

**base** - Runs the baseline experiment.
**cov1** - Runs the covariance split 1 experiment.
**cov2** - Runs the covariance split 2 experiment.
**cov3** - Runs the covariance split 3 experiment.
**cov4** - Runs the covariance split 4 experiment.

If no experiment name is provided, all experiments will run sequentially.

**Example**
- python main.py Content/data/data.csv --experiment base


## Expected Outcomes
Each script will output the performance measures to the console. The execution time for all experiments should not exceed 10 minutes on a standard consumer desktop machine (CPU only). Each individual experiment will take approximately 2 minutes to run (inlcuding cross_validation and hyperparameter optimisation).

The experiment shows that our initial assumption of windowing the dataset to improve model performance was innacurate. The best model is created in Experiment 2 which is using the covariances over the entire dataset (covs_split_1). This model is significantly more accurate (lower log-loss) than the following time splits.

## Dependencies
The project uses numpy, scipy, pandas, argparse, and sklearn libraries as allowed by the coursework specification. The sklearn library is used only for the functionalities specified in Section 1.6 of the coursework brief.

## Reproducibility

For reproducibility, a random seed (SEED = 123)is set within the Gx class in pipeline.py. This ensures that the results are consistent across different runs.

## Reusability 
The code is designed with modularity in mind, facilitating the extension or reuse of functionalities. New evaluation methods can be added without rewriting the model fitting code. Users can explore different parameters of the model by replacing these within the grid search function (pipeline.py).

## Contributions

We welcome contributions! Please let us know if you find anything wrong or have an improvement in mind, by opening an issue or creating a pull request.

## Acknowledgements

We would like to thank Dr Luke Dickens for helping us choose a suitable dataset and providing us with the foundational knowledge to create this project.
