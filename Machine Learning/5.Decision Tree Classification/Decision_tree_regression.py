import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a new sample dataset (e.g., predict salary based on years of experience)
data = {
    'experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'salary':     [25000, 28000, 35000, 40000, 48000, 52000, 60000, 63000, 70000, 75000]
}
df = pd.DataFrame(data)

# Step 2: Split into features and target
X = df[['experience']]  # Feature must be 2D
y = df['salary']

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 5: Predict
y_pred = model.predict(X_test)

# Step 6: Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 7: Visualization
# Predict over a smooth range to draw a curve
X_line = np.arange(0, 11, 0.1).reshape(-1, 1)
y_line_pred = model.predict(X_line)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Data', s=100, edgecolor='k')
plt.plot(X_line, y_line_pred, color='red', label='Decision Tree Prediction', linewidth=2)
plt.title("Decision Tree Regression")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.show()
