# src/preprocessing.py

from typing import List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Builds a preprocessing pipeline that is safe against data leakage.

    - Numerical features are scaled.
    - Categorical features are one-hot encoded.
    - All transformations are fit ONLY on training data
      when used inside an sklearn Pipeline.

    Parameters
    ----------
    numeric_features : List[str]
        Names of numerical columns.
    categorical_features : List[str]
        Names of categorical columns.

    Returns
    -------
    ColumnTransformer
        A preprocessing transformer ready to be used in a pipeline.
    """

    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical_features),
        ],
        remainder="drop"
    )

    return preprocessor


def split_features_target(
    df: pd.DataFrame,
    target_column: str
):
    """
    Separates features and target variable.

    This function does NOT perform any preprocessing
    and should be used BEFORE pipeline fitting.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_column : str
        Name of the target column.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    """

    # Drop rows with missing target values
    df_clean = df.dropna(subset=[target_column])
    
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]

    return X, y


