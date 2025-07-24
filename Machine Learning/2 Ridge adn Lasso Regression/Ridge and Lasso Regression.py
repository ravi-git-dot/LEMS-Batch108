# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score


# load the dataset
california = fetch_california_housing()
dataset =pd.DataFrame(california.data,columns=california.feature_names)
dataset['Price'] =california.target

print(dataset.head())
print(dataset.info())
print(dataset.describe())

dataset_columns =dataset.columns
print(dataset_columns)

#find the null dataset
dataset_nulls =dataset.isnull().sum()
print(dataset_nulls)

#Find the duplicated datasets
dataset_duplicated =dataset.duplicated()
print(dataset_duplicated)

# define the feature and target
X = dataset.drop('Price',axis=1)
y = dataset['Price']

## Linear Regression
lin_regressor = LinearRegression()
mse = cross_val_score(lin_regressor, X, y, scoring='neg_mean_squared_error', cv=5)
mean_mse = -np.mean(mse)
print('Linear_Regression', mean_mse)

## Ridge Regression

ridge = Ridge()
parameters = {'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor = GridSearchCV(ridge,parameters, scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X,y)

print('Best Ridge_regressor alpha',ridge_regressor.best_params_)
print('Best Ridge_regressor score',ridge_regressor.best_score_)

## Lasso regression
lasso = Lasso()
parameters = {'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor = GridSearchCV(lasso,parameters, scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(X,y)

print('Best Lasso alpha',lasso_regressor.best_params_)
print('Best Lasso score',lasso_regressor.best_score_)

## split the data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
prediction_ridge = ridge_regressor.predict(X_test)
prediction_lasso = lasso_regressor.predict(X_test)

print('Prediction Ridge_regressor', prediction_ridge[:5])
print('Prediction Lasso_regressor', prediction_lasso[:5])

## plot the output
sns.histplot(y_test - prediction_ridge, bins=50,kde=True)
plt.title("Ridge Regression -Error Distribution")
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.show()

sns.histplot(y_test - prediction_lasso, bins=50,kde=True)
plt.title("Lasso Regression -Error Distribution")
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.show()
