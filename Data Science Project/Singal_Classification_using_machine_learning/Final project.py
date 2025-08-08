"""
ECG Signal Classification - Final Merged Script
------------------------------------------------
This script loads an ECG dataset, explores it visually, cleans and preprocesses the data,
trains multiple machine learning models, and compares their performance.

Dataset: ECGCvdata.csv
"""

# Step 1: Import the necessary Libraries
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

# Step 2: Load the dataset
df = pd.read_csv("ECGCvdata.csv")
print("\nData Loaded. Shape:", df.shape)
print(df.head())

# Step 3: Dataset information
print("\n--- Data Info ---")
print(df.info())
print("\n--- Description (Numerical Columns) ---")
print(df.describe())

# Step 4: Categorical features, missing values, and duplicates
categorical_cols = [col for col in df.columns if df[col].dtype == 'O']
print("\nCategorical Features:", categorical_cols)
print("Missing values per column:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# Step 5: Target variable inspection
print("\nUnique ECG_signal values:", df['ECG_signal'].unique())
print("\nECG_signal value counts:\n", df['ECG_signal'].value_counts())

# Step 6: Visualizations
main_features = ['hbpermin', 'Pseg', 'QRSseg', 'QTseg']

# Target distribution
sns.countplot(x="ECG_signal", data=df, palette="pastel")
plt.title("ECG Signal Categories")
plt.show()

# Boxplots
for feature in main_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='ECG_signal', y=feature, data=df, palette='rainbow')
    plt.title(f'{feature} vs ECG_signal')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Barplots
for feature in main_features:
    plt.figure(figsize=(6, 4))
    sns.barplot(x='ECG_signal', y=feature, data=df, palette='Set1')
    plt.title(f'{feature} vs ECG_signal')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Missing value heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# Distribution plots
for feature in main_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=feature, kde=True, color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Step 7: Handle missing values
if 'QRtoQSdur' in df.columns:
    df['QRtoQSdur'] = SimpleImputer(strategy='median').fit_transform(df[['QRtoQSdur']])

mean_features = ['RStoQSdur', 'PonPQang', 'PQRang', 'QRSang', 'STToffang',
                 'RSTang', 'QRslope', 'RSslope']
existing_mean_features = [f for f in mean_features if f in df.columns]
df[existing_mean_features] = SimpleImputer(strategy='mean').fit_transform(df[existing_mean_features])

# Step 8: Encode target
label_object = LabelEncoder()
df['ECG_signal'] = label_object.fit_transform(df['ECG_signal'])
print("\nEncoded target classes:", label_object.classes_)

# Step 9: Feature scaling
X = df.drop('ECG_signal', axis=1)
y = df['ECG_signal']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 10: Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Step 11: Train and evaluate multiple models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_object.classes_))

# Step 12: Confusion matrix for Logistic Regression
best_model = LogisticRegression(max_iter=1000)
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_object.classes_, yticklabels=label_object.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: Logistic Regression")
plt.show()

# Step 13: Cross-validation
scores = cross_val_score(best_model, X_scaled, y, cv=5)
print("\nCross-Validation Scores:", scores)
print("Average CV Accuracy:", scores.mean())
