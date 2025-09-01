Project Overview

This project predicts the amount a customer will spend on purchasing a car based on demographic and financial data. Multiple regression models and a neural network are implemented and compared, including Linear Regression, OLS Regression, Random Forest Regressor, and Artificial Neural Network (ANN). The goal is to evaluate model performance and identify the most effective approach for predictive analytics in the automotive sales domain.

Tech Stack

Language: Python 3.x

Libraries:

Data Handling: pandas, numpy

Preprocessing: scikit-learn (StandardScaler, LabelEncoder, train_test_split)

Modeling: scikit-learn (LinearRegression, RandomForestRegressor), statsmodels (OLS), tensorflow/keras (ANN)

Visualization: matplotlib, seaborn

Dataset Description

The dataset contains customer demographic and financial attributes.

Columns (example):

Customer Name – identifier (dropped)

Customer Email – identifier (dropped)

Country – categorical feature (encoded)

Gender – categorical feature (encoded)

Age – numeric feature

Annual Salary – numeric feature

Credit Card Debt – numeric feature

Net Worth – numeric feature

Car Purchase Amount – Target variable

Project Workflow

Data Preprocessing

Load dataset using pandas (Latin-1 encoding for special characters).

Drop irrelevant columns (name, email, redundant pay columns).

Encode categorical variables using LabelEncoder.

Handle skewness with cube root transformation.

Normalize features using StandardScaler.

Exploratory Data Analysis (EDA)

Heatmaps and correlation matrices for feature relationships.

Distribution plots for key variables.

Modeling

Linear Regression (scikit-learn) – baseline model.

OLS Regression (statsmodels) – feature significance analysis.

Random Forest Regressor – ensemble model for non-linear patterns.

Artificial Neural Network (Keras/TensorFlow) – deep learning approach with 3 hidden layers (256-128-64, ReLU activation, Adam optimizer).

Evaluation Metrics

R² Score

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

Results

Best model: Linear Regression (R² ≈ 0.64).

ANN captured non-linear patterns but underperformed due to dataset size.

Random Forest provided competitive performance but less interpretable
