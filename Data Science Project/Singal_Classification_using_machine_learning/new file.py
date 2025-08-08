"""
ECG Signal Classification — Final Submission Script
Merged, cleaned and commented version combining:
- thorough EDA & medical-interpretation notes
- robust preprocessing
- multi-model comparison and evaluation
- cross-validation and model export

Requires:
 - pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
 - dataset file: 'ECGCvdata.csv' in the same directory

How to run:
$ python ECG_signal_classification_final.py

Outputs:
 - console logs of EDA and metrics
 - plots displayed during run
 - saves 'best_model.pkl' (best performing model by CV mean accuracy)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import joblib

RANDOM_STATE = 42

def load_data(path="ECGCvdata.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Put 'ECGCvdata.csv' in this folder.")
    df = pd.read_csv(path)
    print("Data loaded. Shape:", df.shape)
    return df

def quick_info(df):
    print("\n--- Data Info ---")
    print(df.info())
    print("\n--- Numerical summary ---")
    print(df.describe().T)
    print("\nCategorical features:", [c for c in df.columns if df[c].dtype == 'O'])
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

def eda_plots(df):
    sns.set_style("whitegrid")
    # 1) Target counts
    plt.figure(figsize=(7,4))
    sns.countplot(x="ECG_signal", data=df, palette="pastel")
    plt.title("ECG Signal Categories")
    plt.tight_layout()
    plt.show()

    # 2) Boxplots for main clinical features vs target
    main_features = ['hbpermin', 'Pseg', 'QRSseg', 'QTseg']
    for feature in main_features:
        if feature in df.columns:
            plt.figure(figsize=(7,4))
            sns.boxplot(x='ECG_signal', y=feature, data=df)
            plt.title(f'{feature} by ECG_signal')
            plt.tight_layout()
            plt.show()

    # 3) Histograms / distributions
    for feature in main_features:
        if feature in df.columns:
            plt.figure(figsize=(6,4))
            sns.histplot(df[feature].dropna(), kde=True)
            plt.title(f'Distribution of {feature}')
            plt.tight_layout()
            plt.show()

    # 4) Missing-value heatmap
    plt.figure(figsize=(10, 4))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing values (white = missing)")
    plt.tight_layout()
    plt.show()

    # 5) Correlation heatmap (numerical columns only)
    plt.figure(figsize=(10,8))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=False, cmap='coolwarm')
    plt.title("Correlation matrix (numerical features)")
    plt.tight_layout()
    plt.show()

def preprocess(df):
    df = df.copy()
    # Fill missing values using sensible defaults used previously
    # Use median for skewed numeric and mean for others (as in prior scripts)
    # List columns found in prior files that had missing values
    median_cols = [c for c in ['QRtoQSdur'] if c in df.columns]
    mean_cols = [c for c in ['RStoQSdur','PonPQang','PQRang','QRSang','STToffang','RSTang','QRslope','RSslope'] if c in df.columns]

    if median_cols:
        imp_med = SimpleImputer(strategy='median')
        df[median_cols] = imp_med.fit_transform(df[median_cols])
    if mean_cols:
        imp_mean = SimpleImputer(strategy='mean')
        df[mean_cols] = imp_mean.fit_transform(df[mean_cols])

    # If any remaining numeric nulls, fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if df[numeric_cols].isnull().sum().sum() > 0:
        df[numeric_cols] = SimpleImputer(strategy='median').fit_transform(df[numeric_cols])

    # Encode target
    if 'ECG_signal' not in df.columns:
        raise KeyError("Expected target column 'ECG_signal' not found.")
    le = LabelEncoder()
    df['ECG_signal_encoded'] = le.fit_transform(df['ECG_signal'])
    print("\nLabel classes:", list(le.classes_))

    # Features & target
    X = df.drop(columns=['ECG_signal', 'ECG_signal_encoded'], errors='ignore')
    y = df['ECG_signal_encoded']

    # Drop non-numeric features (if any object columns besides ECG_signal remain)
    non_numeric = X.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        print("Dropping non-numeric columns before modeling:", non_numeric)
        X = X.drop(columns=non_numeric)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, le, scaler, X.columns.tolist()

def train_and_evaluate(X, y, feature_names, le, scaler):
    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
    print("\nTrain shape:", x_train.shape, "Test shape:", x_test.shape)

    # Define models to compare
    models = {
        "LogisticRegression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(random_state=RANDOM_STATE, probability=True),
        "KNN": KNeighborsClassifier()
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        print("Classification report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()
        results[name] = {"model": model, "accuracy": acc}

    # Cross-validation on each model (5-fold)
    print("\n--- Cross-validation (5-fold) mean accuracies ---")
    for name, info in results.items():
        model = info["model"]
        cv_scores = cross_val_score(model, X, y, cv=5)
        mean_cv = cv_scores.mean()
        std_cv = cv_scores.std()
        results[name]["cv_mean"] = mean_cv
        results[name]["cv_std"] = std_cv
        print(f"{name}: CV mean = {mean_cv:.4f}, std = {std_cv:.4f}")

    # Select best model by CV mean
    best_name = max(results.keys(), key=lambda k: results[k]["cv_mean"])
    best_info = results[best_name]
    print(f"\nBest model by CV mean accuracy: {best_name} (mean={best_info['cv_mean']:.4f})")

    # Optional: hyperparameter tuning for best model (small grid for demonstration)
    tuned_model = best_info["model"]
    if best_name == "RandomForest":
        param_grid = {"n_estimators":[100,200], "max_depth":[None,10,20]}
    elif best_name == "SVM":
        param_grid = {"C":[0.1,1,10], "kernel":["rbf","linear"]}
    elif best_name == "KNN":
        param_grid = {"n_neighbors":[3,5,7]}
    elif best_name == "LogisticRegression":
        param_grid = {"C":[0.1,1,10], "penalty":["l2"]}  # simple grid
    else:
        param_grid = {}

    if param_grid:
        print(f"\nRunning a small GridSearchCV for {best_name} to slightly improve performance...")
        grid = GridSearchCV(tuned_model, param_grid, cv=4, n_jobs=-1)
        grid.fit(x_train, y_train)
        print("Best params from grid:", grid.best_params_)
        best_model = grid.best_estimator_
    else:
        best_model = tuned_model

    # Final evaluation on test set (using best_model)
    y_test_pred = best_model.predict(x_test)
    final_acc = accuracy_score(y_test, y_test_pred)
    print(f"\nFinal test accuracy ({best_name}): {final_acc:.4f}")
    print("Final classification report:")
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))

    # Save best model + metadata
    model_artifact = {
        "model": best_model,
        "label_encoder": le,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_name": best_name
    }
    joblib.dump(model_artifact, "best_model.pkl")
    print("Saved best model and metadata to 'best_model.pkl'")

    return best_model, results

def main():
    df = load_data()
    quick_info(df)

    # Show EDA plots and insights (kept concise)
    eda_plots(df)

    # Preprocessing
    X_scaled, y, le, scaler, feature_names = preprocess(df)

    # Train & Evaluate
    best_model, results = train_and_evaluate(X_scaled, y, feature_names, le, scaler)

    print("\n--- Quick summary ---")
    sorted_models = sorted(results.items(), key=lambda x: x[1]['cv_mean'], reverse=True)
    for name, info in sorted_models:
        print(f"{name}: CV mean={info['cv_mean']:.4f}, test-accuracy={info['accuracy']:.4f}")

    print("\nAll done. If you want changes (e.g., save all plots, more hyperparameter tuning, or a report notebook), tell me and I'll update the script.")

if __name__ == "__main__":
    main()
