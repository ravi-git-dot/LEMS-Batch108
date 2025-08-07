import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

# Creating the dataset
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "PreviousScores": [50, 55, 60, 63, 65, 70, 75, 80, 85, 90],
    "Pass": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = fail, 1 = pass
}
df = pd.DataFrame(data)

# Feature and target
X = df[["study_hours", "PreviousScores"]]
y = df["Pass"]

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification report:\n", classification_report(y_test, y_pred))

# Visualizing: color-coded predictions vs actual
plt.figure(figsize=(8,5))
scatter = plt.scatter(X_test['study_hours'], X_test['PreviousScores'], c=y_test, cmap='coolwarm', edgecolors='k', s=100)
plt.xlabel('Study Hours')
plt.ylabel('Previous Scores')
plt.title('Pass/Fail Classification by Study Hours and Previous Scores')
plt.grid(True)
plt.colorbar(scatter, label='Pass (1) / Fail (0)')
plt.show()
