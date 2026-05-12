"""Reusable sklearn pipelines for real-estate price prediction."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import MODEL_CATEGORICAL_FEATURES, MODEL_NUMERIC_FEATURES


def get_model_feature_columns() -> list[str]:
    """Return model input columns in stable order."""
    return MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing that imputes/scales numeric data and one-hot encodes categories."""
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


def build_regressor(model_name: str):
    """Create a supported regression estimator by name."""
    regressors = {
        "dummy_median": DummyRegressor(strategy="median"),
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=120,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=160,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=180,
            random_state=42,
            learning_rate=0.06,
            max_depth=3,
            min_samples_leaf=2,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            random_state=42,
            max_iter=160,
            learning_rate=0.08,
            l2_regularization=0.05,
        ),
    }
    if model_name not in regressors:
        supported = ", ".join(sorted(regressors))
        raise ValueError(f"Unknown model '{model_name}'. Supported models: {supported}")
    return regressors[model_name]


def build_model_pipeline(model_name: str) -> Pipeline:
    """Build a complete preprocessing + regression pipeline."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("regressor", build_regressor(model_name)),
        ]
    )
