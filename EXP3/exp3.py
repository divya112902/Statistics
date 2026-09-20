import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances

df = pd.read_csv("diabetes.csv")

print(df.head())

print(df.isnull().sum())

cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']

print((df[cols] == 0).sum())

df[cols] = df[cols].replace(0, np.nan)
print(df.isnull().sum())

imputer = SimpleImputer(strategy='median')

df[cols] = imputer.fit_transform(df[cols])
print(df.isnull().sum())

corr = df.corr()

print(corr)

plt.figure(figsize=(10,8))

sns.heatmap(corr,
            annot=True,
            cmap='coolwarm',
            linewidths=0.5)

plt.title("Correlation Heatmap")
plt.show()

sample = df.iloc[:2,:-1]

print(sample)

euclidean = euclidean_distances(sample)

manhattan = manhattan_distances(sample)

cosine = cosine_distances(sample)

print("Euclidean Distance")
print(euclidean)

print("\nManhattan Distance")
print(manhattan)

print("\nCosine Distance")
print(cosine)

scaler = StandardScaler()

scaled = scaler.fit_transform(df.iloc[:,:-1])

scaled_df = pd.DataFrame(scaled,
                         columns=df.columns[:-1])

print(scaled_df.head())

print(df.iloc[:5,:-1])
print(scaled_df.head())

print(df.describe())
print(scaled_df.describe())