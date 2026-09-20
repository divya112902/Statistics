# EXP 5 – Bootstrap and Permutation Resampling

## Description

- Implements bootstrap and permutation resampling techniques on the Pima Indians Diabetes Dataset.
- Uses bootstrap sampling to estimate the sampling distribution of a statistic.
- Constructs confidence intervals using bootstrap results.
- Uses permutation testing to analyze differences between groups.
- Interprets the resulting distributions and p-values.

## Dataset Used

- Dataset: Pima Indians Diabetes Dataset
- Observations: 768
- Input Attributes: 8

## Requirements

- Python 3.x
- Pandas
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Jupyter Notebook / Google Colab

## Installation

Install the required libraries using:

pip install pandas numpy scipy matplotlib seaborn

## Run

1. Open the EXP 5 Python file or Jupyter Notebook.
2. Load the Pima Indians Diabetes Dataset.
3. Select a numerical variable such as "Glucose" or "BMI".
4. Generate repeated bootstrap samples.
5. Calculate the selected statistic for each sample.
6. Construct the bootstrap confidence interval.
7. Divide the data into groups using "Outcome".
8. Perform the permutation test.
9. Observe and interpret the results.

## Outputs

- Bootstrap sampling distribution
- Bootstrap confidence interval
- Group mean values
- Observed difference between group means
- Permutation distribution
- Permutation p-value
- Statistical interpretation

## Notes

- Bootstrap sampling is performed with replacement.
- Permutation testing randomly shuffles group labels.
- Results may vary slightly between runs because resampling involves randomness.
- A fixed random seed can be used for reproducible results.
- The bootstrap confidence interval is obtained from the bootstrap distribution.