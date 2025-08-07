# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

#  Create a simple dataset (e.g., spatial points)
data = {
    'x': [1, 2, 2, 8, 8, 25, 25, 26, 55, 56, 58, 100],
    'y': [2, 2, 3, 8, 9, 25, 26, 25, 55, 54, 56, 100]
}
df = pd.DataFrame(data)

X = df[['x', 'y']]  # DBSCAN works on 2D or multi-dimensional data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Apply DBSCAN clustering
model = DBSCAN(eps=0.8, min_samples=2)
model.fit(X_scaled)

# Assign cluster labels to the DataFrame
df['cluster'] = model.labels_

# Visualization of clusters
plt.figure(figsize=(8, 6))

# Plot each cluster with different color
unique_labels = set(model.labels_)

for label in unique_labels:
    cluster_points = df[df['cluster'] == label]

    # Label and color logic
    if label == -1:
        color = 'black'
        label_name = 'Noise'
    else:
        color = None
        label_name = f'Cluster {label}'

    plt.scatter(cluster_points['x'], cluster_points['y'],
                label=label_name, s=100, edgecolor='k', c=color)

plt.title("DBSCAN Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
