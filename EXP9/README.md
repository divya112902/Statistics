# EXP 9 – Model Evaluation, Bayesian Concepts and Explainable AI

## Description
- Evaluates machine learning predictions using cross-validation and statistical measures.
- Analyzes the reliability of predicted probabilities using calibration techniques.
- Calculates the Brier Score to evaluate probabilistic predictions.
- Applies Explainable AI (XAI) techniques to understand individual predictions.
- Identifies important features influencing model predictions using SHAP or LIME.

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
- Scikit-learn
- SHAP
- LIME
- Jupyter Notebook / Google Colab

## Installation
Install the required libraries using:

bash
pip install pandas numpy matplotlib scikit-learn shap lime


## Run
1. Open the EXP 9 Python file or Jupyter Notebook.
2. Load the Pima Indians Diabetes Dataset.
3. Preprocess the dataset and handle invalid values.
4. Split the data into training and testing sets.
5. Train a suitable classification model.
6. Perform k-fold cross-validation.
7. Calculate performance metrics such as accuracy and F1-Score.
8. Calculate the Brier Score.
9. Generate a calibration curve.
10. Apply SHAP or LIME to explain selected predictions.
11. Identify important features and interpret the results.

## Outputs
- Trained classification model
- Cross-validation scores
- Mean cross-validation performance
- Accuracy and F1-Score
- Brier Score
- Calibration curve
- SHAP feature importance plot or explanation
- LIME explanation for a selected prediction
- Important features influencing predictions
- Interpretation of model reliability and explainability

## Notes
- Cross-validation provides a more reliable estimate of model performance across different data splits.
- The Brier Score measures the accuracy of predicted probabilities; a lower score is generally better.
- A calibration curve compares predicted probabilities with observed outcome frequencies.
- SHAP explains feature contributions to model predictions.
- LIME explains individual predictions using a locally interpretable approximation.
- Model explanations describe the model's behavior and do not establish medical causation.
- SHAP and LIME may require additional computation depending on the model and dataset.
- Results may vary depending on the model, data preprocessing, and random seed.
