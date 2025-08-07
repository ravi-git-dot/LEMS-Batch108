# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create a simple dataset manually
# Each point has two features: 'income' and 'spending_score'
data = {
    'income': [15, 16, 17, 18, 40, 42, 44, 46, 85, 88, 90, 93],
    'spending_score': [39, 40, 42, 43, 55, 58, 60, 62, 20, 22, 23, 25]
}

# Convert the data into a DataFrame for easier handling
df = pd.DataFrame(data)

#  Select the features we want to use for clustering
# Here we are clustering based on income and spending_score
X = df[['income', 'spending_score']]

#  Create the KMeans model
# We choose 3 clusters for this example (you can change this)
kmeans = KMeans(n_clusters=3, random_state=42)

#  Fit the model on the data
# This means the algorithm will learn and assign clusters
kmeans.fit(X)

# Get the cluster labels for each data point
# These labels will be 0, 1, or 2 depending on which cluster the point belongs to
df['cluster'] = kmeans.labels_

# Step 6: Visualize the clusters using matplotlib
plt.figure(figsize=(8, 6))

# Plot each cluster with a different color
for cluster in range(3):
    # Get all points in this cluster
    cluster_points = df[df['cluster'] == cluster]

    # Plot them
    plt.scatter(cluster_points['income'],
                cluster_points['spending_score'],
                label=f"Cluster {cluster}")

# Step 7: Plot the cluster centers (centroids)
centers = kmeans.cluster_centers_  # Returns coordinates of the 3 centroids
plt.scatter(centers[:, 0], centers[:, 1],
            color='black', marker='X', s=200, label='Centroids')

# Step 8: Label the plot
plt.title("K-Means Clustering Example")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.legend()
plt.grid(True)

# Step 9: Show the plot
plt.show()
