import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df =pd.read_csv('iris.csv')
print(df.head())

print(df.describe())

print(df.shape)

## Univariate Analysis

df_setosa=df.loc[df['Species']=='Iris-setosa']
df_virginica=df.loc[df['Species']=='Iris-virginica']
df_versicolor=df.loc[df['Species']=='Iris-versicolor']

plt.plot(df_setosa['SepalLengthCm'],np.zeros_like(df_setosa['SepalLengthCm']),'o')
plt.plot(df_versicolor['SepalLengthCm'],np.zeros_like(df_versicolor['SepalLengthCm']),'o')
plt.plot(df_virginica['SepalLengthCm'],np.zeros_like(df_virginica['SepalLengthCm']),'o')
plt.xlabel('Petal Length (cm)')
plt.show()

## Bivariate Analysis

sns.FacetGrid(df, hue = 'Species', aspect=0.5, height=4, legend_out=True).\
    map(plt.scatter, 'PetalLengthCm', 'SepalLengthCm').add_legend()
plt.show()

'''
Clear Separation by Species:
   1. Setosa forms a distinct cluster, clearly separated from the other two species.
   2.Versicolor and Virginica are closer to each other but still show some separation.
Positive Correlation:
   1.There is a positive trend: as PetalLengthCm increases, SepalLengthCm also increases.
Cluster Behavior:
   1.Setosa: Smaller petal and sepal lengths.
   2.Virginica: Larger petal and sepal lengths.
   3.Versicolor: Intermediate values.

'''

### Multivariate Analysis

sns.pairplot(df, hue='Species', height=5, diag_kind='kde')
plt.show()