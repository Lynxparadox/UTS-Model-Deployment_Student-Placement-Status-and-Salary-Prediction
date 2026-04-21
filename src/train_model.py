from pathlib import Path
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import pickle

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor

from src.data_ingestion import ingest_data
from src.preprocessing import preprocess_data
from src.evaluate import evaluate_classification, evaluate_regression
from src.pipeline import create_pipeline

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR.parent / "artifacts_"

best_lr_params = {
    "C": 1.2,
    "penalty": "l2",
    "solver": "liblinear",
    "max_iter": 1000
}

def train_model():

    # MLflow setup
    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    mlflow.set_experiment("Placement Status and Salary Prediction")

    # Load data
    df = ingest_data()

    # Preprocess data
    X, y_class, y_reg = preprocess_data()
    
    # Split the data
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Pipeline
    print("Building pipeline...")
    # Optimized Logistic Regression for classification
    class_model = create_pipeline(
        LogisticRegression(**best_lr_params),
        X_train
    )
    # LGBMRegressor for regression
    reg_model = create_pipeline(
        LGBMRegressor(random_state=42),
        X_train
    )

    # Train and MLflow
    with mlflow.start_run() as run:

        class_model.fit(X_train, y_class_train)
        reg_model.fit(X_train, y_reg_train)

        # Evluate and log metrics
        f1 = evaluate_classification(class_model, X_test, y_class_test)
        rmse = evaluate_regression(reg_model, X_test, y_reg_test)   

        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_params(best_lr_params)

        mlflow.sklearn.log_model(sk_model=class_model, name="classification_model")
        mlflow.sklearn.log_model(sk_model=reg_model, name="regression_model")

        print(f'F1 Score : {f1:.4f}')
        print(f'RMSE : {rmse:.4f}')
        print(f"Model trained and logged to MLflow with Run ID: {run.info.run_id}")

        # Save the models
        os.makedirs("artifacts", exist_ok=True)

        with open(ARTIFACT_DIR / "classification_model.pkl", "wb") as f:
            pickle.dump(class_model, f)

        with open(ARTIFACT_DIR / "regression_model.pkl", "wb") as f:
            pickle.dump(reg_model, f)

        print(f"Models saved to successfully to {ARTIFACT_DIR}")

if __name__ == "__main__":
    train_model()