# ============================================================
# EXPERIMENT: BOOTSTRAP CONFIDENCE INTERVAL & PERMUTATION TEST
# Dataset: Pima Indians Diabetes Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Load the Pima Indians Diabetes Dataset
# ------------------------------------------------------------

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

df = pd.read_csv(url, header=None, names=columns)

print("========== DATASET INFORMATION ==========")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ------------------------------------------------------------
# 2. Select numerical variable: Glucose
# ------------------------------------------------------------

glucose = df["Glucose"].values

print("\n========== ORIGINAL DATA ==========")
print("Sample size:", len(glucose))
print("Original mean Glucose:", round(np.mean(glucose), 3))


# ------------------------------------------------------------
# 3 & 4. Bootstrap Sampling
# ------------------------------------------------------------

np.random.seed(42)

B = 5000
n = len(glucose)

bootstrap_means = []

for i in range(B):

    # Draw sample with replacement
    sample = np.random.choice(
        glucose,
        size=n,
        replace=True
    )

    # Calculate mean
    bootstrap_means.append(np.mean(sample))

bootstrap_means = np.array(bootstrap_means)


# ------------------------------------------------------------
# 5. Construct 95% Bootstrap Confidence Interval
# ------------------------------------------------------------

lower = np.percentile(bootstrap_means, 2.5)
upper = np.percentile(bootstrap_means, 97.5)

bootstrap_mean = np.mean(bootstrap_means)
bootstrap_se = np.std(bootstrap_means, ddof=1)

print("\n========== BOOTSTRAP RESULTS ==========")
print("Number of bootstrap samples:", B)
print("Bootstrap mean:", round(bootstrap_mean, 3))
print("Bootstrap standard error:", round(bootstrap_se, 3))
print(
    "95% Bootstrap Confidence Interval:",
    (round(lower, 3), round(upper, 3))
)


# ------------------------------------------------------------
# DIAGRAM 1: Bootstrap Sampling Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    bootstrap_means,
    bins=50,
    edgecolor="black"
)

plt.axvline(
    lower,
    linestyle="--",
    label="Lower 2.5%"
)

plt.axvline(
    upper,
    linestyle="--",
    label="Upper 97.5%"
)

plt.xlabel("Mean Glucose")
plt.ylabel("Frequency")
plt.title("Bootstrap Sampling Distribution of Mean Glucose")
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 6. Divide data into two groups based on Outcome
# ------------------------------------------------------------

group0 = df[df["Outcome"] == 0]["Glucose"].values
group1 = df[df["Outcome"] == 1]["Glucose"].values

print("\n========== GROUP INFORMATION ==========")
print("Outcome 0 (No Diabetes):", len(group0))
print("Outcome 1 (Diabetes):", len(group1))


# ------------------------------------------------------------
# 7. Calculate observed difference between group means
# ------------------------------------------------------------

mean0 = np.mean(group0)
mean1 = np.mean(group1)

observed_difference = mean1 - mean0

print("\n========== GROUP MEANS ==========")
print("Mean Glucose - Outcome 0:", round(mean0, 3))
print("Mean Glucose - Outcome 1:", round(mean1, 3))
print(
    "Observed Difference (1 - 0):",
    round(observed_difference, 3)
)


# ------------------------------------------------------------
# DIAGRAM 2: Glucose by Outcome
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

data = [group0, group1]

plt.boxplot(
    data,
    label=["0 (No Diabetes)", "1 (Diabetes)"]
)

plt.xlabel("Outcome")
plt.ylabel("Glucose")
plt.title("Glucose by Outcome")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 8 & 9. Permutation Test
# ------------------------------------------------------------

np.random.seed(42)

B_perm = 10000

combined = df["Glucose"].values
labels = df["Outcome"].values

permutation_differences = []

for i in range(B_perm):

    # Randomly shuffle outcome labels
    shuffled_labels = np.random.permutation(labels)

    # Create groups using shuffled labels
    perm_group0 = combined[shuffled_labels == 0]
    perm_group1 = combined[shuffled_labels == 1]

    # Difference between means
    difference = (
        np.mean(perm_group1)
        - np.mean(perm_group0)
    )

    permutation_differences.append(difference)

permutation_differences = np.array(
    permutation_differences
)


# ------------------------------------------------------------
# 10. Compare observed difference with permutation distribution
# ------------------------------------------------------------

p_value = np.mean(
    np.abs(permutation_differences)
    >= abs(observed_difference)
)

print("\n========== PERMUTATION TEST ==========")
print("Number of permutations:", B_perm)
print(
    "Observed Difference:",
    round(observed_difference, 3)
)

if p_value == 0:
    print("Permutation Test p-value: < 0.0001")
else:
    print(
        "Permutation Test p-value:",
        round(p_value, 4)
    )

if p_value < 0.05:

    print("\nDecision: Reject the Null Hypothesis.")
    print(
        "Conclusion: There is a statistically significant"
    )
    print(
        "difference in mean Glucose between the two Outcome groups."
    )

else:

    print("\nDecision: Fail to Reject the Null Hypothesis.")
    print(
        "Conclusion: There is no statistically significant"
    )
    print(
        "difference in mean Glucose between the two Outcome groups."
    )


# ------------------------------------------------------------
# DIAGRAM 3: Permutation Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    permutation_differences,
    bins=50,
    edgecolor="black"
)

plt.axvline(
    observed_difference,
    linestyle="--",
    linewidth=2,
    label="Observed Difference"
)

plt.axvline(
    -observed_difference,
    linestyle="--",
    linewidth=2,
    label="Negative Observed Difference"
)

plt.xlabel("Difference in Mean Glucose")
plt.ylabel("Frequency")
plt.title("Permutation Distribution of Difference in Mean Glucose")
plt.legend()
plt.tight_layout()
plt.show()