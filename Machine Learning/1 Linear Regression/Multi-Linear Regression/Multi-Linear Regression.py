
# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

#Convert the column into categorical columns
states=pd.get_dummies(X['State'],drop_first=True)

# Drop the state coulmn
X=X.drop('State',axis=1)

# concat the dummy variables
X=pd.concat([X,states],axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predict the model
y_pred = regressor.predict(X_test)
print(y_pred)

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)

# Use R&D Spend (usually the first column) for visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_test['R&D Spend'], y_test, color='blue', label='Actual Profit')
plt.scatter(X_test['R&D Spend'], y_pred, color='green', edgecolor='black', label='Predicted Profit')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.title('Actual vs Predicted Profit based on R&D Spend')
plt.legend()
plt.grid(True)
plt.show()
