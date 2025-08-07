import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create simple dataset
data = {
    'Feature1': [2, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    'Feature2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Split data into features and target
X = df[['Feature1', 'Feature2']]
y = df['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train)

# Create and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Simple visualization
plt.scatter(X_test['Feature1'], X_test['Feature2'], c=y_test, cmap='coolwarm', edgecolors='k')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('Logistic Regression Classification')
plt.show()