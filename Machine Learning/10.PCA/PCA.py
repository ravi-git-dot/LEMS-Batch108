# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Create a simple dataset with 3 features
data = {
    'math_score':     [70, 80, 85, 90, 95, 65, 60, 75],
    'science_score':  [72, 78, 88, 94, 96, 66, 62, 73],
    'english_score':  [65, 70, 78, 85, 90, 60, 58, 68]
}
df = pd.DataFrame(data)

#Standardize the features (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

#Apply PCA to reduce from 3D to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#Create a new DataFrame with principal components
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])

#Visualize the principal components
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], color='teal', s=100, edgecolor='k')
plt.title("PCA Result (3D → 2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

#Print explained variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
