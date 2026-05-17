"""Training and artifact creation for the real-estate Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data_cleaning import build_cleaned_dataset, create_cleaning_summary
from src.evaluation import compare_regression_results, evaluate_regression_predictions
from src.features import add_model_features
from src.model_pipeline import build_model_pipeline, get_model_feature_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "nekretnine_raw.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
MODEL_ARTIFACT_NAME = "real_estate_price_pipeline.joblib"
MODEL_METADATA_NAME = "model_metadata.json"

DEFAULT_MODEL_CANDIDATES: list[tuple[str, str]] = [
    ("Baseline - DummyRegressor median", "dummy_median"),
    ("LinearRegression", "linear_regression"),
    ("RidgeRegression", "ridge_regression"),
    ("RandomForestRegressor", "random_forest"),
    ("ExtraTreesRegressor", "extra_trees"),
    ("GradientBoostingRegressor", "gradient_boosting"),
    ("HistGradientBoostingRegressor", "hist_gradient_boosting"),
]


def _json_safe_number(value: Any) -> Any:
    """Convert pandas/numpy scalar values into JSON-safe Python values."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _category_options(series: pd.Series, limit: int | None = None) -> list[str]:
    """Return stable category options ordered by frequency then label."""
    counts = series.dropna().astype(str).value_counts()
    if limit is not None:
        counts = counts.head(limit)
    return sorted(counts.index.tolist())


def _portable_project_path(path: str | Path) -> str:
    """Store repository-local paths relatively and external paths as provided."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def collect_ui_options(model_df: pd.DataFrame) -> dict[str, Any]:
    """Collect category choices and numeric ranges for Streamlit controls."""
    regions_by_city = {
        str(city): _category_options(city_df["region"])
        for city, city_df in model_df.dropna(subset=["city"]).groupby("city", sort=True)
    }
    options: dict[str, Any] = {
        "cities": _category_options(model_df["city"]),
        "regions": _category_options(model_df["region"]),
        "regions_by_city": regions_by_city,
        "heating_types": _category_options(model_df["heating_type"]),
        "parking_options": _category_options(model_df["parking"]),
    }

    numeric_ranges = {}
    for column in ["area_m2", "rooms", "year_built", "price_eur"]:
        values = model_df[column].dropna()
        numeric_ranges[column] = {
            "min": _json_safe_number(values.min()),
            "median": _json_safe_number(values.median()),
            "max": _json_safe_number(values.max()),
        }
    options["numeric_ranges"] = numeric_ranges
    return options


def prepare_modeling_data(
    data_path: str | Path = DEFAULT_DATA_PATH,
    current_year: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load raw data, clean it, add model features, and return summary metadata."""
    raw_df = pd.read_csv(data_path)
    cleaned_df = build_cleaned_dataset(raw_df)
    model_df = add_model_features(cleaned_df, current_year=current_year)
    cleaning_summary = create_cleaning_summary(raw_df)
    return raw_df, model_df, cleaning_summary


def train_candidate_models(
    model_df: pd.DataFrame,
    candidate_specs: list[tuple[str, str]] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Train candidate regressors and return fitted models, metrics, and split data."""
    specs = candidate_specs or DEFAULT_MODEL_CANDIDATES
    feature_columns = get_model_feature_columns()
    X = model_df[feature_columns]
    y = model_df["price_eur"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    fitted_models = {}
    results = []
    progress_specs = tqdm(
        specs,
        desc="Training model candidates",
        unit="model",
        disable=not show_progress,
    )
    for display_name, model_key in progress_specs:
        pipeline = build_model_pipeline(model_key)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        fitted_models[display_name] = {
            "model_key": model_key,
            "pipeline": pipeline,
        }
        results.append(evaluate_regression_predictions(display_name, y_test, predictions))

    results_table = compare_regression_results(results)
    best_model_name = str(results_table.iloc[0]["model"])
    best_model_key = fitted_models[best_model_name]["model_key"]

    return {
        "fitted_models": fitted_models,
        "metrics_table": results_table,
        "best_model_name": best_model_name,
        "best_model_key": best_model_key,
        "best_pipeline": fitted_models[best_model_name]["pipeline"],
        "feature_columns": feature_columns,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def build_model_metadata(
    training_result: dict[str, Any],
    model_df: pd.DataFrame,
    cleaning_summary: dict[str, Any],
    artifact_path: Path,
    data_path: str | Path,
    random_state: int,
    test_size: float,
) -> dict[str, Any]:
    """Build JSON-serializable metadata for the saved model artifact."""
    metrics = training_result["metrics_table"].to_dict(orient="records")
    metrics = [
        {key: _json_safe_number(value) for key, value in row.items()}
        for row in metrics
    ]

    return {
        "best_model_name": training_result["best_model_name"],
        "best_model_key": training_result["best_model_key"],
        "artifact_path": _portable_project_path(artifact_path),
        "data_path": _portable_project_path(data_path),
        "feature_columns": training_result["feature_columns"],
        "metrics": metrics,
        "raw_rows": int(cleaning_summary["raw_rows"]),
        "cleaned_rows": int(len(model_df)),
        "train_rows": int(len(training_result["X_train"])),
        "test_rows": int(len(training_result["X_test"])),
        "test_size": test_size,
        "random_state": random_state,
        "ui_options": collect_ui_options(model_df),
        "cleaning_summary": {
            key: _json_safe_number(value) for key, value in cleaning_summary.items()
        },
    }


def train_and_save_best_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    candidate_specs: list[tuple[str, str]] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    current_year: int = 2026,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Train all candidates, save the best pipeline, and write model metadata."""
    data_path = Path(data_path)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    _, model_df, cleaning_summary = prepare_modeling_data(
        data_path=data_path,
        current_year=current_year,
    )
    training_result = train_candidate_models(
        model_df=model_df,
        candidate_specs=candidate_specs,
        test_size=test_size,
        random_state=random_state,
        show_progress=show_progress,
    )

    artifact_path = models_dir / MODEL_ARTIFACT_NAME
    metadata_path = models_dir / MODEL_METADATA_NAME
    joblib.dump(training_result["best_pipeline"], artifact_path, compress=3)

    metadata = build_model_metadata(
        training_result=training_result,
        model_df=model_df,
        cleaning_summary=cleaning_summary,
        artifact_path=artifact_path,
        data_path=data_path,
        random_state=random_state,
        test_size=test_size,
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "artifact_path": str(artifact_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
        "metrics_table": training_result["metrics_table"],
        "best_pipeline": training_result["best_pipeline"],
        "X_test": training_result["X_test"],
        "y_test": training_result["y_test"],
    }


def main() -> None:
    """CLI entrypoint used before running the Streamlit app."""
    result = train_and_save_best_model(show_progress=True)
    metadata = result["metadata"]
    print(f"Saved model artifact: {result['artifact_path']}")
    print(f"Saved model metadata: {result['metadata_path']}")
    print(f"Best model by MAE: {metadata['best_model_name']}")
    print(result["metrics_table"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
