# Titanic Survival Prediction

## Introduction

This repository contains a Python script for predicting the survival of passengers on the Titanic using machine learning techniques. The dataset used for this analysis is the famous Titanic dataset, which contains information about passengers on board the Titanic, including whether they survived or not.

## Dependencies

Before running the script, make sure you have the following dependencies installed:
- Python (3.x recommended)
- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost (optional, for XGBoost)

You can install these dependencies using pip by running:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Running the Script

To run the script, follow these steps:

1. Clone this repository to your local machine or download the script `titanic_survival_prediction.py`.

2. Open a terminal or command prompt and navigate to the directory containing the script.

3. Run the script using Python:

```bash
python titanic_survival_prediction.py
```

4. The script will load the Titanic dataset, perform data preprocessing, exploratory data analysis (EDA), and train various machine learning models. Finally, it will display the accuracy of each model and provide insights into survival predictions.
   

## Machine Learning Models

The script uses several machine learning models for prediction, including:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes
- XGBoost (optional, if you have it installed)

## Cross-Validation

The script performs k-fold cross-validation to evaluate the models' performance and prevent overfitting. It displays the average cross-validated accuracy for each model.

## Hyperparameter Tuning

Hyperparameter tuning is performed for some models, such as SVM and Random Forest, to find the best parameters for maximizing accuracy.

## Ensembling

The script demonstrates ensembling techniques like Voting Classifier, Bagging, and Boosting to combine the strengths of multiple models for better prediction accuracy.

## Results

The script provides insights into the performance of each model, helping you choose the best model for Titanic survival prediction.
