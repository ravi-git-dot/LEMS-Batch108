import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create a simple dataset
data = {
    'age': [22, 25, 47, 52, 46, 56, 48, 55, 60, 35],
    'salary': [15000, 29000, 48000, 60000, 52000, 61000, 58000, 57000, 65000, 40000],
    'purchased': [0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  # 0 = No, 1 = Yes
}

df = pd.DataFrame(data)

# Step 2: Define features and target
X = df[['age', 'salary']]
y = df['purchased']

# Step 3: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Create and train Random Forest model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Plot decision boundary
plt.figure(figsize=(10, 6))

# Scatter plot of actual data
plt.scatter(X['age'], X['salary'], c=y, edgecolor='k', cmap=plt.cm.Paired, s=100)
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Random Forest Classifier Decision Boundary")
plt.grid(True)
plt.show()