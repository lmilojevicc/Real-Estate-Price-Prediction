"""Reusable sklearn pipelines for real-estate price prediction."""

from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
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


LOG_TARGET_SUFFIX = "_log_target"


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
            n_estimators=800,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=1,
            max_features=0.8,
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
        "catboost": CatBoostRegressor(
            iterations=700,
            learning_rate=0.05,
            depth=6,
            loss_function="MAE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            cat_features=tuple(MODEL_CATEGORICAL_FEATURES),
        ),
        "xgboost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=9,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            eval_metric="mae",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        ),
    }
    if model_name not in regressors:
        supported = ", ".join(sorted(regressors))
        raise ValueError(f"Unknown model '{model_name}'. Supported models: {supported}")
    return regressors[model_name]


def build_model_pipeline(model_name: str) -> Pipeline:
    """Build a complete preprocessing + regression pipeline."""
    use_log_target = model_name.endswith(LOG_TARGET_SUFFIX)
    base_model_name = model_name.removesuffix(LOG_TARGET_SUFFIX)
    regressor = build_regressor(base_model_name)

    if base_model_name == "catboost":
        estimator = regressor
    else:
        estimator = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("regressor", regressor),
            ]
        )

    if use_log_target:
        return TransformedTargetRegressor(
            regressor=estimator,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )

    return estimator
