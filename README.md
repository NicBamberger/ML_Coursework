# Multivariate Gait Data Classification

## Overview

This repository contains the code and documentation for a time series Machine Learning project focused on classifying lowerbody bracing conditions using Multivariate Gait Data. The project utilizes the **Gradient Boosting Classifier** within an ensemble learning framework for predictive modeling.

## Project Structure

- **data**: This directory holds the Multivariate Gait Dataset obtained from the UC Irvine Machine Learning Repository (https://archive.ics.uci.edu/dataset/760/multivariate+gait+data).
- **notebooks**: This repository includes 3 files:

1. **`pipeline.py`**
   - This file defines the core functionalities of the machine learning pipeline, encapsulated within the `Gx` class. Key functions include data processing, feature generation, cross-validation, grid search, and model training.

2. **`experiments.py`**
   - This file orchestrates the entire Machine Learning process. It uses functions from `pipeline.py` to load data, generate feature sets, perform grid search, choose the best model, and test the results. The `main()` function ties everything together for cohesive execution.

3. **`tests.ipynb`**
   - This Jupyter Notebook is designed for interactive testing and experimentation. It loads a small dataset, processes the data using the `Gx` class, creates feature sets, performs grid search, and tests the resulting model. It also includes visualizations to aid in understanding the data and results.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy`


## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/multivariate-gait-classification.git
   cd multivariate-gait-classification
   ```

2. Set up a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

This will execute the machine learning pipeline, performing data processing, feature generation, grid search, and model evaluation.

4. Explore the Jupyter notebooks in the `notebooks` directory for data analysis and model development.

## Usage

1. **Run the Experiment:**

   ```bash
   python experiments.py
   ```

   This will execute the machine learning pipeline, performing data processing, feature generation, grid search, and model evaluation.

2. **Interactive Testing (Optional):**

   Open and run the Jupyter Notebook `tests.ipynb` to interactively test and experiment with the pipeline on a small dataset.

## Contributions

We welcome contributions! Please let us know if you find anything wrong or have an improvement in mind, by opening an issue or creating a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
```

Remember to include a `LICENSE` file in your project directory if you choose a license for your project. You can replace the `LICENSE` link in the README with a link to your actual license file.

Feel free to customize the README according to your specific project details and structure.
```
## Acknowledgements

We would like to thank Dr Luke Dickens for helping us choose a suitable dataset and providing us with the foundational knowledge to create this project.

---
