import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from ucimlrepo import fetch_ucirepo


# Function to fetch and preprocess the Iris dataset
def fetch_and_load_iris_dataset():
    iris = fetch_ucirepo(id=53)
    X = iris.data.features
    y = iris.data.targets
    return X.values, y.values.ravel()  # Convert to numpy arrays and flatten y to 1D


# Discretize the features (if necessary)
def discretize_features(X, n_bins=10):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    X_discrete = discretizer.fit_transform(X)
    return X_discrete


# Function to calculate mutual information between features and target
def calculate_mutual_information(X, y):
    mi_scores = []
    for i in range(X.shape[1]):
        mi = mutual_info_score(X[:, i], y)
        mi_scores.append(mi)
    return mi_scores


# Function to generate synthetic datasets and calculate bias
def investigate_bias(original_X, original_y, steps=5):
    n_samples, n_features = original_X.shape
    original_mi = calculate_mutual_information(original_X, original_y)

    all_mi_scores = []
    sample_sizes = np.linspace(n_samples, n_samples * 10, steps, dtype=float)  # Ensure float type

    for size in sample_sizes:
        train_size = min(float(size / n_samples), 0.99)  # Ensure train_size is < 1.0
        synthetic_X, _, synthetic_y, _ = train_test_split(original_X, original_y, train_size=train_size,
                                                          random_state=42)
        mi_scores = calculate_mutual_information(synthetic_X, synthetic_y)
        all_mi_scores.append((size, mi_scores))

    return original_mi, all_mi_scores


# Example
if __name__ == "__main__":
    original_X, original_y = fetch_and_load_iris_dataset()

    # Discretize features if they are continuous
    original_X = discretize_features(original_X, n_bins=10)

    original_mi, all_mi_scores = investigate_bias(original_X, original_y)

    print("Original Mutual Information Scores:", original_mi)
    for size, mi_scores in all_mi_scores:
        print(f"Sample Size: {size}, Mutual Information Scores: {mi_scores}")
