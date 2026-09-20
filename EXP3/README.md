# EXP 3 – Correlation, Distance Measures and Data Preprocessing

## Description

- Performs correlation analysis on the Pima Indians Diabetes Dataset.
- Measures similarity and dissimilarity between selected observations.
- Handles selected missing or invalid values.
- Applies data normalization or standardization.
- Compares data before and after preprocessing.

## Dataset Used

- Dataset: Pima Indians Diabetes Dataset
- Observations: 768
- Input Attributes: 8
- Target Variable: "Outcome"
- "Outcome = 0" → Non-diabetic
- "Outcome = 1" → Diabetic

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

pip install pandas numpy matplotlib seaborn scikit-learn

## Run

1. Open the EXP 3 Python file or Jupyter Notebook.
2. Load the Pima Indians Diabetes Dataset.
3. Identify and handle invalid values.
4. Run the correlation and distance calculations.
5. Apply the selected preprocessing technique.
6. View the results and visualizations.

## Outputs

- Correlation matrix
- Correlation heatmap
- Euclidean distance
- Manhattan distance
- Cosine distance
- Preprocessed data
- Normalized/standardized values
- Comparison of data before and after preprocessing

## Notes

- Pearson correlation is used to measure linear relationships.
- Distance calculations are performed between selected observations.
- Scaling is important for distance-based methods.
- Median imputation can be useful when extreme values are present.