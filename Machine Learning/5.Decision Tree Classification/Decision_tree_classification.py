import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Create a new simple dataset
data = {
    'experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'interview_score': [40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
    'hired': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  # 0 = Not Hired, 1 = Hired
}
df = pd.DataFrame(data)

# Features and target
X = df[['experience', 'interview_score']]
y = df['hired']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train the Decision Tree model
model = DecisionTreeClassifier(random_state=1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualization (Actual vs Predicted)
plt.figure(figsize=(12, 5))

# Subplot 1: Actual labels ---
plt.subplot(1, 2, 1)
plt.scatter(X_test['experience'], X_test['interview_score'], c=y_test, cmap='coolwarm', s=100, edgecolor='k')
plt.title("Actual Labels")
plt.xlabel("Experience")
plt.ylabel("Interview Score")
plt.grid(True)

# Subplot 2: Predicted labels ---
plt.subplot(1, 2, 2)
plt.scatter(X_test['experience'], X_test['interview_score'], c=y_pred, cmap='coolwarm', s=100, edgecolor='k')
plt.title("Predicted Labels by Decision Tree")
plt.xlabel("Experience")
plt.ylabel("Interview Score")
plt.grid(True)

plt.tight_layout()
plt.show()
