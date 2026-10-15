# EXP 8 – Clustering and Dimensionality Reduction

## Description
- Applies unsupervised learning techniques to the Pima Indians Diabetes Dataset.
- Discovers hidden patterns and groups similar patient observations.
- Uses K-Means clustering to group observations.
- Selects a suitable number of clusters using the Elbow Method.
- Evaluates cluster quality using the Silhouette Score.
- Applies Principal Component Analysis (PCA) to reduce dimensionality and visualize the data.

## Dataset Used
- **Dataset:** Pima Indians Diabetes Dataset
- **Observations:** 768
- **Input Attributes:** 8
- **Target Variable:** Outcome
- Outcome = 0 → Non-diabetic
- Outcome = 1 → Diabetic

**Note:** The `Outcome` variable is excluded during clustering and may be used afterward to compare the discovered clusters with known diabetes categories.

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
1. Open the EXP 8 Python file or Jupyter Notebook.
2. Load the Pima Indians Diabetes Dataset.
3. Select the relevant numerical features.
4. Handle missing or invalid values.
5. Standardize the selected features.
6. Apply K-Means clustering.
7. Use the Elbow Method to select a suitable number of clusters.
8. Calculate the Silhouette Score.
9. Apply PCA to reduce the data to two dimensions.
10. Visualize and interpret the clusters.

## Outputs
- Preprocessed and standardized dataset
- Elbow plot
- Selected number of clusters
- Cluster labels
- Cluster centroids
- Silhouette Score
- PCA-transformed data
- Two-dimensional cluster visualization
- Interpretation of discovered patterns

## Notes
- Clustering is performed without using the `Outcome` variable.
- Feature standardization is important because K-Means uses distance calculations.
- The Elbow Method helps identify a suitable number of clusters.
- A higher Silhouette Score generally indicates better-separated clusters.
- PCA reduces the number of dimensions while preserving as much variance as possible.
- Discovered clusters do not necessarily correspond directly to diabetic and non-diabetic categories.
