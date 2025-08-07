#Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

#Create a different sample dataset
data = {
    'income': [15, 16, 17, 40, 42, 44, 85, 88, 90],
    'spending_score': [39, 40, 41, 55, 57, 60, 20, 22, 25]
}
df = pd.DataFrame(data)

# Prepare feature matrix
X = df[['income', 'spending_score']]

# Plot the dendrogram using scipy
linked = linkage(X, method='ward')  # 'ward' minimizes variance within clusters

plt.figure(figsize=(10, 6))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title("Dendrogram (Hierarchical Clustering)")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

# Step 4: Apply Agglomerative Clustering (choose 3 clusters based on dendrogram)
model = AgglomerativeClustering(n_clusters=3,linkage='ward')
df['cluster'] = model.fit_predict(X)

# Step 5: Visualize clusters
plt.figure(figsize=(8, 6))
for cluster in df['cluster'].unique():
    clustered_data = df[df['cluster'] == cluster]
    plt.scatter(clustered_data['income'], clustered_data['spending_score'], label=f'Cluster {cluster}', s=100)

plt.title("Hierarchical Clustering (Agglomerative)")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.legend()
plt.grid(True)
plt.show()
