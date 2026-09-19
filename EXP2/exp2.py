import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("diabetes.csv")
df.head()

#load dataset

print("========== FIRST 5 ROWS ==========")
print(df.head())

#inspect dataset

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== DATASET INFO ==========")
df.info()

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

#descriptive statistics

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())

print("\nMean")
print(df.mean(numeric_only=True))

print("\nMedian")
print(df.median(numeric_only=True))

print("\nMode")
print(df.mode().iloc[0])

print("\nMinimum")
print(df.min(numeric_only=True))

print("\nMaximum")
print(df.max(numeric_only=True))

print("\nVariance")
print(df.var(numeric_only=True))

print("\nStandard Deviation")
print(df.std(numeric_only=True))

#histograms

df.hist(figsize=(12,10))
plt.suptitle("Histogram of Numerical Variables")
plt.show()

for col in df.columns[:-1]:
    plt.figure(figsize=(5,4))
    plt.boxplot(df[col])
    plt.title("Boxplot of " + col)
    plt.ylabel(col)
    plt.show()

outcome = df["Outcome"].value_counts().sort_index()

plt.figure(figsize=(6,4))
plt.bar(["0 (Non-Diabetic)", "1 (Diabetic)"], outcome)
plt.title("Outcome Frequency")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df["Glucose"], df["BMI"])
plt.title("Glucose vs BMI")
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df["Age"], df["Glucose"])
plt.title("Age vs Glucose")
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.show()

pd.plotting.scatter_matrix(
    df[['Pregnancies','Glucose','BMI','Age']],
    figsize=(10,10),
    diagonal='hist'
)

plt.suptitle("Pair Plot")
plt.show()


print("\n========== OBSERVATIONS ==========")
print("1. Dataset contains", df.shape[0], "rows and", df.shape[1], "columns.")
print("2. Outcome has two classes: 0 (Non-Diabetic) and 1 (Diabetic).")
print("3. Histograms show the distribution of numerical variables.")
print("4. Boxplots reveal outliers in Glucose, BMI and Insulin.")
print("5. The bar chart shows more Non-Diabetic than Diabetic patients.")
print("6. Glucose and BMI show a weak positive relationship.")
print("7. Age and Glucose do not show a strong linear relationship.")
print("8. Pair plot helps visualize relationships among numerical variables.")
print("9. Glucose is an important feature for diabetes prediction.")
print("10. The dataset is suitable for machine learning classification.")












