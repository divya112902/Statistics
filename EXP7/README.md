# EXP 7 – Diabetes Classification

## Description
- Develops supervised classification models to predict diabetes outcomes using the Pima Indians Diabetes Dataset.
- Classifies patients as diabetic or non-diabetic.
- Handles missing or invalid values and preprocesses the data.
- Trains and evaluates classification models.
- Compares model performance using appropriate classification metrics.

## Dataset Used
- **Dataset:** Pima Indians Diabetes Dataset
- **Observations:** 768
- **Input Attributes:** 8
- **Target Variable:** Outcome
- Outcome = 0 → Non-diabetic
- Outcome = 1 → Diabetic

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
1. Open the EXP 7 Python file or Jupyter Notebook.
2. Load the Pima Indians Diabetes Dataset.
3. Separate the input features and target variable.
4. Identify and handle invalid or missing values.
5. Split the dataset using stratified sampling.
6. Apply feature scaling where required.
7. Train the selected classification model(s).
8. Generate predictions for the test data.
9. Construct the confusion matrix.
10. Calculate and interpret the classification metrics.

## Outputs
- Trained classification model(s)
- Predicted diabetes outcomes
- Confusion matrix
- Accuracy
- Precision
- Recall
- F1-Score
- ROC curve
- ROC-AUC score
- Comparison of classification models
- Interpretation of model performance

## Notes
- Logistic Regression, k-Nearest Neighbors (KNN), and Decision Tree can be used as classification models.
- Feature scaling is particularly important for models such as KNN.
- A stratified train-test split helps preserve class proportions.
- Precision and recall are important when evaluating diabetes classification.
- ROC-AUC measures the model's ability to distinguish between the two classes.
- Predictions are intended for educational purposes and not for medical diagnosis.
