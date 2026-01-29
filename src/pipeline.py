# src/pipeline.py

from typing import List
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.linear_model import LinearRegression

try:
    from .preprocessing import build_preprocessor
except ImportError:
    from preprocessing import build_preprocessor


def build_regression_pipeline(
    numeric_features: List[str],
    categorical_features: List[str]
) -> SKPipeline:
    """
    Builds an end-to-end regression pipeline:
    preprocessing + linear regression.

    This pipeline guarantees:
    - No data leakage
    - Consistent preprocessing across splits
    - Reproducible training

    Parameters
    ----------
    numeric_features : List[str]
        Names of numerical columns.
    categorical_features : List[str]
        Names of categorical columns.

    Returns
    -------
    Pipeline
        sklearn Pipeline object ready for fit/predict.
    """

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    model = LinearRegression()

    pipeline = SKPipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    return pipeline
