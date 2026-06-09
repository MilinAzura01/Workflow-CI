"""
modelling.py untuk MLProject - Workflow CI
"""
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import os

mlflow.set_experiment("Titanic-Classification")

TRAIN_PATH = "titanic_preprocessing/train.csv"
TEST_PATH  = "titanic_preprocessing/test.csv"

def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    X_train = train.drop("Survived", axis=1)
    y_train = train["Survived"]
    X_test  = test.drop("Survived", axis=1)
    y_test  = test["Survived"]
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_data()
    
    # Menggunakan autolog dari MLflow
    mlflow.autolog()
    
    with mlflow.start_run(run_name="RandomForest-Basic"):
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Mengukur performa pada test set agar metrik evaluasi tercatat otomatis oleh autolog
        score = model.score(X_test, y_test)
        print(f"[INFO] Model evaluation score: {score:.4f}")
        
    print("[INFO] Model trained, parameters, metrics, and artifacts are logged automatically via MLflow Autolog.")

if __name__ == "__main__":
    main()