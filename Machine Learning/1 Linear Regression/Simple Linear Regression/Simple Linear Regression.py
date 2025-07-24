# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# sample datasets

data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [35000, 37000, 40000, 43000, 45000, 48000, 50000, 55000, 60000, 63000]
}

df = pd.DataFrame(data)

# Step 1: Define feature and target
X = df[['Experience']]  # Feature
y = df['Salary']        # Target

# Step 2 : Split train and Test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3 create a train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4 make prediction
y_pred = model.predict(X_test)

# Step 5 Evaluate the model
# Mean Squared Error: Shows the average squared difference between predicted and actual values.
# R2 Score: Measures how well the regression line fits the data (1 is perfect fit).

print('Mean_squared_error',mean_squared_error(y_test, y_pred))
print('R^2',r2_score(y_test, y_pred))

# Step 6 Plot the result

# Visualization: Full Data with Regression Line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.scatter(X_test, y_pred, color='green', s=45, edgecolor='black', label='Predicted (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Actual vs Predicted Salary')
plt.legend()
plt.grid(True)
plt.show()
