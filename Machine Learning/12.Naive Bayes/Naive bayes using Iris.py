# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data                  # Features (4 columns)
y = iris.target                # Target labels (0, 1, 2)
target_names = iris.target_names

# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

# Step 5: Reduce to 2D with PCA for visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Predict on all data for visualization
y_all_pred = model.predict(X)

# Step 6: Plot actual vs predicted in PCA space
plt.figure(figsize=(12, 5))

# --- Actual labels ---
plt.subplot(1, 2, 1)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='Set1', edgecolor='k', s=100)
plt.title("Actual Iris Classes")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)

# --- Predicted labels ---
plt.subplot(1, 2, 2)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_all_pred, cmap='Set1', edgecolor='k', s=100)
plt.title("Predicted by Naive Bayes")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)

plt.tight_layout()
plt.show()
