"""Reusable preprocessing helpers for Activity 3 and later modeling."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import MODEL_CATEGORICAL_FEATURES, MODEL_NUMERIC_FEATURES


def get_model_feature_columns() -> list[str]:
    """Return model input columns in stable order, excluding the target price."""
    return MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing for numeric and categorical real-estate features."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Nepoznato")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, MODEL_NUMERIC_FEATURES),
            ("cat", categorical_pipeline, MODEL_CATEGORICAL_FEATURES),
        ]
    )
