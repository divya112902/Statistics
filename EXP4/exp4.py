# ============================================================
# STATISTICAL ANALYSIS OF PIMA INDIANS DIABETES DATASET
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats

# ------------------------------------------------------------
# 1. LOAD THE DATASET
# ------------------------------------------------------------

print("=" * 70)
print("STATISTICAL ANALYSIS OF PIMA INDIANS DIABETES DATASET")
print("=" * 70)

# Load dataset
df = pd.read_csv("diabetes.csv")

print("\n1. DATASET LOADED SUCCESSFULLY")
print("-" * 70)

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nFirst 5 rows:")
print(df.head())


# ------------------------------------------------------------
# 2. BASIC INFORMATION
# ------------------------------------------------------------

print("\n\n2. DATASET INFORMATION")
print("-" * 70)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 3. SELECT APPROPRIATE VARIABLES
# ------------------------------------------------------------

# We will use:
# Glucose -> numerical variable
# BMI -> numerical variable
# Age -> numerical variable
# Outcome -> categorical/binary variable
#
# Outcome:
# 0 = Non-diabetic
# 1 = Diabetic

print("\n\n3. SELECTED VARIABLES")
print("-" * 70)

print("Numerical variables: Glucose, BMI, Age")
print("Categorical variable: Outcome")
print("Outcome: 0 = Non-diabetic, 1 = Diabetic")


# ------------------------------------------------------------
# 4. DIVIDE DATA INTO SUITABLE GROUPS
# ------------------------------------------------------------

non_diabetic = df[df["Outcome"] == 0]
diabetic = df[df["Outcome"] == 1]

print("\n\n4. GROUPING THE DATA")
print("-" * 70)

print("Non-diabetic group:", len(non_diabetic))
print("Diabetic group:", len(diabetic))


# ============================================================
# PART A: POINT ESTIMATES
# ============================================================

print("\n\n" + "=" * 70)
print("PART A: POINT ESTIMATES")
print("=" * 70)


# ------------------------------------------------------------
# 5. SAMPLE MEAN
# ------------------------------------------------------------

glucose_mean = df["Glucose"].mean()
bmi_mean = df["BMI"].mean()
age_mean = df["Age"].mean()

print("\n5. SAMPLE MEANS")
print("-" * 70)

print(f"Mean Glucose = {glucose_mean:.2f}")
print(f"Mean BMI     = {bmi_mean:.2f}")
print(f"Mean Age     = {age_mean:.2f}")


# ------------------------------------------------------------
# 6. SAMPLE PROPORTION
# ------------------------------------------------------------

diabetic_count = df["Outcome"].sum()
total_count = len(df)

sample_proportion = diabetic_count / total_count

print("\n6. SAMPLE PROPORTION")
print("-" * 70)

print("Number of diabetic individuals:", diabetic_count)
print("Total individuals:", total_count)
print(f"Sample proportion of diabetic individuals = {sample_proportion:.4f}")
print(f"Sample proportion (%) = {sample_proportion * 100:.2f}%")


# ============================================================
# PART B: CONFIDENCE INTERVALS
# ============================================================

print("\n\n" + "=" * 70)
print("PART B: CONFIDENCE INTERVALS")
print("=" * 70)


# ------------------------------------------------------------
# 7. 95% CONFIDENCE INTERVAL FOR MEAN GLUCOSE
# ------------------------------------------------------------

sample = df["Glucose"].dropna()

n = len(sample)
mean = sample.mean()
std = sample.std(ddof=1)

confidence_level = 0.95
alpha = 1 - confidence_level

# t critical value
t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

margin_error = t_critical * (std / np.sqrt(n))

lower_mean = mean - margin_error
upper_mean = mean + margin_error

print("\n7. 95% CONFIDENCE INTERVAL FOR MEAN GLUCOSE")
print("-" * 70)

print(f"Sample mean = {mean:.2f}")
print(f"Sample standard deviation = {std:.2f}")
print(f"Sample size = {n}")
print(f"Margin of error = {margin_error:.2f}")

print(f"95% Confidence Interval = ({lower_mean:.2f}, {upper_mean:.2f})")

print(
    f"\nInterpretation: We are 95% confident that the population "
    f"mean glucose level lies between {lower_mean:.2f} and "
    f"{upper_mean:.2f}."
)


# ------------------------------------------------------------
# 8. 95% CONFIDENCE INTERVAL FOR DIABETES PROPORTION
# ------------------------------------------------------------

p = sample_proportion
n = total_count

z_critical = stats.norm.ppf(1 - alpha / 2)

margin_proportion = z_critical * np.sqrt((p * (1 - p)) / n)

lower_prop = p - margin_proportion
upper_prop = p + margin_proportion

print("\n\n8. 95% CONFIDENCE INTERVAL FOR DIABETES PROPORTION")
print("-" * 70)

print(f"Sample proportion = {p:.4f}")
print(f"Margin of error = {margin_proportion:.4f}")

print(
    f"95% Confidence Interval = "
    f"({lower_prop:.4f}, {upper_prop:.4f})"
)

print(
    f"\nIn percentage terms: "
    f"({lower_prop * 100:.2f}%, {upper_prop * 100:.2f}%)"
)

print(
    f"\nInterpretation: We are 95% confident that the true "
    f"population proportion of diabetic individuals lies between "
    f"{lower_prop * 100:.2f}% and {upper_prop * 100:.2f}%."
)


# ============================================================
# PART C: ONE-SAMPLE T-TEST
# ============================================================

print("\n\n" + "=" * 70)
print("PART C: ONE-SAMPLE T-TEST")
print("=" * 70)


# Research question:
# Is the average glucose level significantly different from 120?

# H0: μ = 120
# H1: μ != 120

hypothesized_mean = 120

t_stat, p_value = stats.ttest_1samp(
    df["Glucose"].dropna(),
    hypothesized_mean
)

print("\n9. ONE-SAMPLE T-TEST")
print("-" * 70)

print("Research Question:")
print("Is the population mean glucose level significantly different from 120?")

print("\nNull Hypothesis (H0):")
print("The population mean glucose level is equal to 120.")

print("\nAlternative Hypothesis (H1):")
print("The population mean glucose level is not equal to 120.")

print("\nSignificance level (alpha) = 0.05")

print(f"\nSample mean = {df['Glucose'].mean():.2f}")
print(f"Test statistic (t) = {t_stat:.4f}")
print(f"p-value = {p_value:.6f}")

if p_value < 0.05:
    print("\nDecision: Reject H0.")
    print(
        "Interpretation: There is sufficient statistical evidence "
        "to conclude that the population mean glucose level is "
        "significantly different from 120."
    )
else:
    print("\nDecision: Fail to reject H0.")
    print(
        "Interpretation: There is insufficient statistical evidence "
        "to conclude that the population mean glucose level is "
        "significantly different from 120."
    )


# ============================================================
# PART D: TWO-SAMPLE T-TEST
# ============================================================

print("\n\n" + "=" * 70)
print("PART D: TWO-SAMPLE T-TEST")
print("=" * 70)


# Research question:
# Do diabetic and non-diabetic individuals have different
# average glucose levels?

glucose_non_diabetic = non_diabetic["Glucose"].dropna()
glucose_diabetic = diabetic["Glucose"].dropna()

# H0: μ1 = μ2
# H1: μ1 != μ2

t_stat_two, p_value_two = stats.ttest_ind(
    glucose_diabetic,
    glucose_non_diabetic,
    equal_var=False
)

print("\n10. TWO-SAMPLE T-TEST")
print("-" * 70)

print("Research Question:")
print(
    "Is there a significant difference in mean glucose levels "
    "between diabetic and non-diabetic individuals?"
)

print("\nNull Hypothesis (H0):")
print(
    "There is no significant difference in mean glucose levels "
    "between the two groups."
)

print("\nAlternative Hypothesis (H1):")
print(
    "There is a significant difference in mean glucose levels "
    "between the two groups."
)

print("\nSignificance level (alpha) = 0.05")

print(
    f"\nMean glucose - Diabetic group = "
    f"{glucose_diabetic.mean():.2f}"
)

print(
    f"Mean glucose - Non-diabetic group = "
    f"{glucose_non_diabetic.mean():.2f}"
)

print(f"\nTest statistic (t) = {t_stat_two:.4f}")
print(f"p-value = {p_value_two:.6f}")

if p_value_two < 0.05:
    print("\nDecision: Reject H0.")
    print(
        "Interpretation: There is a statistically significant "
        "difference in mean glucose levels between diabetic and "
        "non-diabetic individuals."
    )
else:
    print("\nDecision: Fail to reject H0.")
    print(
        "Interpretation: There is insufficient evidence of a "
        "significant difference in mean glucose levels between "
        "the two groups."
    )


# ============================================================
# PART E: CHI-SQUARE TEST
# ============================================================

print("\n\n" + "=" * 70)
print("PART E: CHI-SQUARE TEST")
print("=" * 70)


# Research question:
# Is there an association between age group and diabetes outcome?

# Create age groups

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 30, 50, 100],
    labels=["Young (<=30)", "Middle (31-50)", "Older (>50)"]
)

# Create contingency table

contingency_table = pd.crosstab(
    df["AgeGroup"],
    df["Outcome"]
)

# Rename columns
contingency_table.columns = [
    "Non-Diabetic",
    "Diabetic"
]

print("\n11. CHI-SQUARE TEST")
print("-" * 70)

print("\nContingency Table:")
print(contingency_table)

# H0: Age group and diabetes status are independent
# H1: Age group and diabetes status are associated

chi2, p_chi, dof, expected = stats.chi2_contingency(
    contingency_table
)

print("\nResearch Question:")
print(
    "Is there a significant association between age group "
    "and diabetes status?"
)

print("\nNull Hypothesis (H0):")
print(
    "Age group and diabetes status are independent."
)

print("\nAlternative Hypothesis (H1):")
print(
    "Age group and diabetes status are associated."
)

print("\nSignificance level (alpha) = 0.05")

print(f"\nChi-square test statistic = {chi2:.4f}")
print(f"Degrees of freedom = {dof}")
print(f"p-value = {p_chi:.6f}")

print("\nExpected Frequencies:")
print(
    pd.DataFrame(
        expected,
        index=contingency_table.index,
        columns=contingency_table.columns
    ).round(2)
)

if p_chi < 0.05:
    print("\nDecision: Reject H0.")
    print(
        "Interpretation: There is a statistically significant "
        "association between age group and diabetes status."
    )
else:
    print("\nDecision: Fail to reject H0.")
    print(
        "Interpretation: There is insufficient statistical evidence "
        "of an association between age group and diabetes status."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n\n" + "=" * 70)
print("FINAL SUMMARY OF RESULTS")
print("=" * 70)

print("\nPoint Estimates:")
print(f"Mean Glucose = {glucose_mean:.2f}")
print(f"Mean BMI = {bmi_mean:.2f}")
print(f"Mean Age = {age_mean:.2f}")
print(f"Diabetes Proportion = {sample_proportion:.4f}")

print("\nConfidence Intervals:")
print(
    f"95% CI for Mean Glucose = "
    f"({lower_mean:.2f}, {upper_mean:.2f})"
)

print(
    f"95% CI for Diabetes Proportion = "
    f"({lower_prop:.4f}, {upper_prop:.4f})"
)

print("\nHypothesis Tests:")

print(
    f"\n1. One-Sample t-test:"
    f"\n   t-statistic = {t_stat:.4f}"
    f"\n   p-value = {p_value:.6f}"
)

if p_value < 0.05:
    print("   Decision = Reject H0")
else:
    print("   Decision = Fail to reject H0")


print(
    f"\n2. Two-Sample t-test:"
    f"\n   t-statistic = {t_stat_two:.4f}"
    f"\n   p-value = {p_value_two:.6f}"
)

if p_value_two < 0.05:
    print("   Decision = Reject H0")
else:
    print("   Decision = Fail to reject H0")


print(
    f"\n3. Chi-square test:"
    f"\n   Chi-square = {chi2:.4f}"
    f"\n   p-value = {p_chi:.6f}"
)

if p_chi < 0.05:
    print("   Decision = Reject H0")
else:
    print("   Decision = Fail to reject H0")


print("\n" + "=" * 70)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)