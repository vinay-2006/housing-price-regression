# src/evaluation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(y_true, y_pred) -> dict:
    """
    Computes standard regression metrics.

    Returns a dictionary for clean logging and comparison.
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def baseline_predictors(y_train, strategy: str = "mean"):
    """
    Generates trivial baseline predictions.

    Parameters
    ----------
    y_train : array-like
        Training target values.
    strategy : str
        'mean' or 'median'

    Returns
    -------
    float
        Baseline prediction value.
    """

    if strategy == "mean":
        return y_train.mean()
    elif strategy == "median":
        return y_train.median()
    else:
        raise ValueError("strategy must be 'mean' or 'median'")


def plot_residuals(y_true, y_pred):
    """
    Plots residual diagnostics:
    - Residuals vs predictions
    - Residual distribution
    """

    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs Predictions
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(0, linestyle="--")
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predictions")

    # Residual Distribution
    axes[1].hist(residuals, bins=30)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")

    plt.tight_layout()
    plt.show()


def detect_outliers(residuals, threshold: float = 3.0):
    """
    Simple outlier detection using z-score on residuals.

    Returns indices of extreme residuals.
    """

    z_scores = (residuals - residuals.mean()) / residuals.std()
    return np.where(np.abs(z_scores) > threshold)[0]

def get_residuals(y_true, y_pred):
    """
    Returns residuals for downstream analysis.
    """
    return y_true - y_pred
