"""
modelling_tuning.py
Hyperparameter Tuning + Manual Logging + DagsHub
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    ConfusionMatrixDisplay, classification_report
)

# ─── Konfigurasi DagsHub ───
dagshub.init(
    repo_owner='MilinAzura01',
    repo_name='Eksperimen_SML_MilinAzura',
    mlflow=True
)
mlflow.set_experiment("Titanic-Tuning")


def load_data():
    train = pd.read_csv("titanic_preprocessing/train.csv")
    test  = pd.read_csv("titanic_preprocessing/test.csv")
    X_train = train.drop("Survived", axis=1)
    y_train = train["Survived"]
    X_test  = test.drop("Survived", axis=1)
    y_test  = test["Survived"]
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=['Not Survived', 'Survived']).plot(ax=ax)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100)
    plt.close()


def plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=100)
    plt.close()


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    plt.figure(figsize=(8, 5))
    plt.barh(sorted_features, importances[indices], color='steelblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=100)
    plt.close()


def main():
    X_train, X_test, y_train, y_test = load_data()
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5],
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model  = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"[INFO] Best params: {best_params}")

    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')

    with mlflow.start_run(run_name="RandomForest-Tuning-ManualLog"):

        # Log params
        mlflow.log_param("model_type", "RandomForestClassifier")
        for k, v in best_params.items():
            mlflow.log_param(k, v)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("cv_f1_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_std", cv_scores.std())
        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        # Log model
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # Artefak tambahan
        plot_confusion_matrix(y_test, y_pred)
        mlflow.log_artifact("confusion_matrix.png")

        plot_roc_curve(y_test, y_proba)
        mlflow.log_artifact("roc_curve.png")

        plot_feature_importance(best_model, list(X_train.columns))
        mlflow.log_artifact("feature_importance.png")

        report = classification_report(y_test, y_pred, output_dict=True)
        with open("classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact("classification_report.json")

        print(f"\n[RESULTS]")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")

    print("\n[DONE] Semua artefak tersimpan di DagsHub!")


if __name__ == "__main__":
    main()