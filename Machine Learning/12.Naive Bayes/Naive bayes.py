# Naive Bayes Classification
# Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#Create a simple dataset (e.g., based on age and salary to predict buying behavior)
data = {
    'age': [22, 25, 47, 52, 46, 56, 48, 55, 60, 35],
    'salary': [15000, 29000, 48000, 60000, 52000, 61000, 58000, 57000, 65000, 40000],
    'purchased': [0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  # 0 = Not Purchased, 1 = Purchased
}
df = pd.DataFrame(data)

# Split into features and target
X = df[['age', 'salary']]
y = df['purchased']

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

#Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Visualization (Actual vs Predicted)
plt.figure(figsize=(12, 5))

# Actual labels
plt.subplot(1, 2, 1)
plt.scatter(X_test['age'], X_test['salary'], c=y_test, cmap='coolwarm', s=100, edgecolor='k')
plt.title("Actual Labels")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.grid(True)

# Predicted labels
plt.subplot(1, 2, 2)
plt.scatter(X_test['age'], X_test['salary'], c=y_pred, cmap='coolwarm', s=100, edgecolor='k')
plt.title("Predicted Labels by Naive Bayes")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.grid(True)

plt.tight_layout()
plt.show()
