# ECG Signal Classification Project
# Cleaned and Enhanced Version with Improvements

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load Dataset
df = pd.read_csv("ECGCvdata.csv")
print("Data Loaded. Shape:", df.shape)

# Step 3: Basic Info
print(df.info())
print(df.describe())
print("Categorical Features:", [col for col in df.columns if df[col].dtype == 'O'])
print("Missing Values:", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# Step 4: Visualizations - ECG Signal Counts
sns.countplot(x="ECG_signal", data=df, palette="pastel")
plt.title("ECG Signal Categories")
plt.show()

# Step 5: Boxplot Visualization
main_features = ['hbpermin', 'Pseg', 'QRSseg', 'QTseg']
for feature in main_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='ECG_signal', y=feature, data=df)
    plt.title(f'{feature} vs ECG_signal')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Step 6: Impute Missing Values
df['QRtoQSdur'] = SimpleImputer(strategy='median').fit_transform(df[['QRtoQSdur']])
features_mean = ['RStoQSdur', 'PonPQang', 'PQRang', 'QRSang', 'STToffang', 'RSTang', 'QRslope', 'RSslope']
df[features_mean] = SimpleImputer(strategy='mean').fit_transform(df[features_mean])

# Step 7: Label Encode Target
le = LabelEncoder()
df['ECG_signal'] = le.fit_transform(df['ECG_signal'])
print("Encoded Labels:", le.classes_)

# Step 8: Feature Scaling
X = df.drop('ECG_signal', axis=1)
y = df['ECG_signal']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 9: Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 10: Model Training & Comparison
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"\n{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

# Step 11: Confusion Matrix for Best Model (example: Logistic Regression)
best_model = LogisticRegression()
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: Logistic Regression")
plt.show()

# Step 12: Cross Validation
scores = cross_val_score(best_model, X_scaled, y, cv=5)
print("Cross-Validation Scores:", scores)
print("Average CV Accuracy:", scores.mean())
