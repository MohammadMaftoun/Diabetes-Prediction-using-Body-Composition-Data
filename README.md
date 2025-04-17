# Diabetes Body Composition Data

This repository contains Python scripts for analyzing the relationship between body composition and diabetes using machine learning. The code is divided into two main scripts: preprocessing.py for data preparation and postprocessing.py for model training, evaluation, and visualization.
Features

# Data Preprocessing (preprocessing.py):
Loads and explores body composition data from an Excel file.
Handles missing values, converts data types, and scales features using MinMaxScaler.
Generates synthetic data using CTGAN to address class imbalance.
Saves preprocessed data to a CSV file.


# Model Training and Evaluation (postprocessing.py):
Trains multiple classifiers: MLP, Gradient Boosting, Random Forest, Logistic Regression, Decision Tree, LightGBM, XGBoost, AdaBoost, LDA, and TabNet.
Evaluates models using metrics like AUROC, accuracy, precision, recall, F1-score, and more.
Visualizes the ROC curve for the MLP model and a correlation heatmap for feature analysis.


# Dependencies:
pandas, numpy, scikit-learn, ctgan, pytorch-tabnet, mealpy, xgboost, lightgbm, plotly, matplotlib



# Usage

Run preprocessing.py to clean, scale, and augment the data.
Run postprocessing.py to train models, evaluate performance, and visualize results.

# Requirements

Python 3.10+
Install dependencies: pip install pandas numpy scikit-learn ctgan pytorch-tabnet mealpy xgboost lightgbm plotly matplotlib

