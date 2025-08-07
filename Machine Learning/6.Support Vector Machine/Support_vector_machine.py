import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create a new sample dataset
data = {
    'hours_studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'sleep_hours':   [9, 8, 7, 6, 5, 5, 4, 3, 2, 1],
    'pass_exam':     [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}
df = pd.DataFrame(data)

# Step 2: Features and target
X = df[['hours_studied', 'sleep_hours']]
y = df['pass_exam']

# Step 3: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Step 4: Train the SVM model
model = SVC(kernel='linear')  # You can also try 'rbf', 'poly', etc.
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#  Visualization
# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X['hours_studied'], X['sleep_hours'], c=y, edgecolor='k', s=100)
plt.xlabel("Hours Studied")
plt.ylabel("Sleep Hours")
plt.title("SVM Decision Boundary")
plt.grid(True)
plt.show()
