import numpy as np
from sklearn.metrics import f1_score, mean_squared_error

def evaluate_classification(model, X, y):
    preds = model.predict(X)
    f1 = f1_score(y, preds)

    print("Classification Evaluation")
    print(f"\nF1 Score: {f1:.4f}")

    return f1

def evaluate_regression(model, X, y):
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))

    print("\nRegression Evaluation")
    print(f"\nRMSE: {rmse:.4f}")

    return rmse