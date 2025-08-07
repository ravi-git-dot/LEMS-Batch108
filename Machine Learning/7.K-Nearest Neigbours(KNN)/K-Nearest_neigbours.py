import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create a different sample dataset
data = {
    'age': [18, 20, 22, 25, 30, 35, 40, 45, 50, 60],
    'exercise_hours': [1, 2, 2, 3, 4, 3, 2, 1, 1, 0],
    'fit': [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  # 0 = Not Fit, 1 = Fit
}
df = pd.DataFrame(data)

# Step 2: Features and target
X = df[['age', 'exercise_hours']]
y = df['fit']

# Step 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Visualization (actual vs predicted)
plt.figure(figsize=(12, 5))

# Actual labels
plt.subplot(1, 2, 1)
plt.scatter(X_test['age'], X_test['exercise_hours'], c=y_test, cmap='coolwarm', s=100, edgecolor='k')
plt.title("Actual Labels")
plt.xlabel("Age")
plt.ylabel("Exercise Hours")
plt.grid(True)

# Predicted labels
plt.subplot(1, 2, 2)
plt.scatter(X_test['age'], X_test['exercise_hours'], c=y_pred, cmap='coolwarm', s=100, edgecolor='k')
plt.title("Predicted Labels by KNN")
plt.xlabel("Age")
plt.ylabel("Exercise Hours")
plt.grid(True)

plt.tight_layout()
plt.show()
