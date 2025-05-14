# Unsupervised-Learning-Techniques-PCA-K-Means-Clustering-and-Data-Validation

This project is a practical implementation of two essential unsupervised learning techniques — Principal Component Analysis (PCA) for dimensionality reduction and K-Means clustering for unsupervised classification — along with utilities for data validation and visualization.

## Overview

The goal of this assignment is to understand and apply:
- **Principal Component Analysis (PCA)**: Reduce the number of features in a dataset while retaining as much variance as possible.
- **K-Means Clustering**: Group data points into clusters based on similarity.
- **Validation & Visualization**: Analyze clustering performance and visualize feature-reduced data.

## Repository Contents

```
.
├── classifiers.py
├── feature_reduction.py
├── PCA__K_Means__and_Data_Validation.ipynb
└── README.md
```

### Files

- **`classifiers.py`**
  - Implements a base `Classifier` class and a custom `KMeans` class.
  - Includes centroid initialization, iterative clustering, and Euclidean distance calculations.

- **`feature_reduction.py`**
  - Defines a base `FeatureReduction` class and the `PrincipleComponentAnalysis` subclass.
  - Implements PCA with threshold-based variance retention and projection matrix calculation.

- **`PCA__K_Means__and_Data_Validation.ipynb`**
  - Notebook for combining feature reduction and clustering, and visualizing the results.
  - Contains code for data normalization, PCA fit/predict, and K-Means clustering workflow.

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/unsupervised-learning-techniques-pca-kmeans-clustering.git
cd pca-kmeans-clustering
```

### 2. Set Up the Environment

We recommend using a virtual environment.

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, manually install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 3. Run the Notebook

Use Jupyter Notebook or an IDE to open and execute:

```bash
jupyter notebook HW3__PCA__K_Means__and_Data_Validation.ipynb
```

## Functionality

### Principal Component Analysis (PCA)

- Standardizes input features.
- Computes the covariance matrix and its eigen decomposition.
- Retains principal components based on cumulative variance threshold (default: 95%).

### K-Means Clustering

- Randomly initializes `k` cluster centroids.
- Iteratively assigns points to the nearest centroid and updates centroids.
- Predicts the nearest cluster for new data points.

### Visualization

- 2D scatter plots of PCA-reduced data.
- Optional display of K-Means clustering results.

## Example Usage

```python
from feature_reduction import PrincipleComponentAnalysis
from classifiers import KMeans
import pandas as pd

# Load your data as a NumPy array
# data = ...

# Apply PCA
pca = PrincipleComponentAnalysis()
pca.fit(data)
reduced_data = pca.predict(data)

# Apply K-Means
kmeans = KMeans()
kmeans.fit(reduced_data, k=3)
```

## License

This project is intended for educational and experimental use.
