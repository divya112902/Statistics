# EXP 6 – Regression Analysis

## Description
- Develops regression models to predict a continuous health-related variable using the Pima Indians Diabetes Dataset.
- Uses medical attributes as input features to predict BMI.
- Preprocesses missing or invalid values.
- Trains and tests a regression model.
- Evaluates model performance using statistical metrics.
- Analyzes residuals and compares actual and predicted values.

## Dataset Used
- **Dataset:** Pima Indians Diabetes Dataset
- **Observations:** 768
- **Input Attributes:** 8
- **Target Variable:** BMI (continuous variable)
- **Predictor Variables:** Relevant medical attributes such as Pregnancies, Glucose, BloodPressure, Insulin, and Age.

## Requirements
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / Google Colab

## Installation
Install the required libraries using:

bash
pip install pandas numpy matplotlib seaborn scikit-learn


## Run
1. Open the EXP 6 Python file or Jupyter Notebook.
2. Load the Pima Indians Diabetes Dataset.
3. Select the input features and continuous target variable.
4. Handle missing or invalid values.
5. Split the dataset into training and testing sets.
6. Train the regression model.
7. Generate predictions for the test data.
8. Calculate the evaluation metrics.
9. Analyze residuals and compare actual versus predicted values.

## Outputs
- Training and testing datasets
- Regression model predictions
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Actual versus predicted values
- Residual analysis plots
- Interpretation of regression performance

## Notes
- BMI is selected as the continuous target variable.
- Invalid or missing values should be handled before model training.
- MAE, MSE, and RMSE measure prediction errors.
- R² Score indicates how much variation in the target variable is explained by the model.
- Residuals are calculated as actual values minus predicted values.
- Regression predictions are for educational purposes and should not be used for medical diagnosis.
