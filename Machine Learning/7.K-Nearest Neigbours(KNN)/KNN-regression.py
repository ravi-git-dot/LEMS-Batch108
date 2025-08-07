import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a simple dataset (e.g., predict test score based on study hours)
data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'test_score':  [30, 35, 40, 45, 50, 55, 65, 70, 78, 85]
}
df = pd.DataFrame(data)

# Step 2: Split into features and target
X = df[['study_hours']]  # Feature needs to be 2D
y = df['test_score']

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train KNN Regressor
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 6: Visualization (Regression curve)
X_line = np.linspace(0, 11, 100).reshape(-1, 1)
y_line_pred = model.predict(X_line)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Data', s=100, edgecolor='k')
plt.plot(X_line, y_line_pred, color='red', label='KNN Prediction', linewidth=2)
plt.title("KNN Regression")
plt.xlabel("Study Hours")
plt.ylabel("Test Score")
plt.legend()
plt.grid(True)
plt.show()
