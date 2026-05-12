"""Prediction helpers used by the Streamlit real-estate app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features import add_model_features
from src.model_pipeline import get_model_feature_columns
from src.training import DEFAULT_MODELS_DIR, MODEL_ARTIFACT_NAME, MODEL_METADATA_NAME


class PropertyInputValidationError(ValueError):
    """Raised when app-style property input cannot be used for prediction."""


def _is_missing(value: Any) -> bool:
    """Return True for None/NaN values from forms or DataFrames."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _optional_float(value: Any) -> float:
    """Convert optional numeric form values to float or NaN."""
    if _is_missing(value) or value == "":
        return np.nan
    return float(value)


def validate_property_input(property_input: dict[str, Any]) -> None:
    """Validate property values against the same broad ranges used for modeling."""
    errors = []

    for field in ["city", "region", "heating_type", "parking"]:
        if not str(property_input.get(field, "")).strip():
            errors.append(f"{field} is required")

    try:
        area_m2 = float(property_input.get("area_m2"))
        if not 10 <= area_m2 <= 500:
            errors.append("area_m2 must be between 10 and 500")
    except (TypeError, ValueError):
        errors.append("area_m2 must be a number between 10 and 500")

    rooms = property_input.get("rooms")
    if not _is_missing(rooms) and rooms != "":
        try:
            rooms_value = float(rooms)
            if not 0.5 <= rooms_value <= 10:
                errors.append("rooms must be between 0.5 and 10")
        except (TypeError, ValueError):
            errors.append("rooms must be a number between 0.5 and 10")

    year_built = property_input.get("year_built")
    if not _is_missing(year_built) and year_built != "":
        try:
            year_value = float(year_built)
            if not 1800 <= year_value <= 2028:
                errors.append("year_built must be between 1800 and 2028")
        except (TypeError, ValueError):
            errors.append("year_built must be a number between 1800 and 2028")

    floor = property_input.get("floor")
    total_floors = property_input.get("total_floors")
    if not _is_missing(floor) and floor != "":
        try:
            floor_value = float(floor)
            if not -5 <= floor_value <= 200:
                errors.append("floor must be between -5 and 200")
        except (TypeError, ValueError):
            errors.append("floor must be a number")
    if not _is_missing(total_floors) and total_floors != "":
        try:
            total_floors_value = float(total_floors)
            if not 0 <= total_floors_value <= 200:
                errors.append("total_floors must be between 0 and 200")
        except (TypeError, ValueError):
            errors.append("total_floors must be a number")

    if errors:
        raise PropertyInputValidationError("; ".join(errors))


def _format_number(value: float) -> str:
    """Format floor numbers without unnecessary decimals."""
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def build_raw_floor_string(floor: Any, total_floors: Any) -> str:
    """Convert app floor inputs into the raw floor string parsed by feature engineering."""
    floor_value = _optional_float(floor)
    total_value = _optional_float(total_floors)

    if np.isnan(floor_value):
        return "Nepoznato"

    if floor_value < 0:
        floor_label = "Suteren"
    elif floor_value == 0:
        floor_label = "Prizemlje"
    else:
        floor_label = _format_number(floor_value)

    if np.isnan(total_value):
        return floor_label
    return f"{floor_label} / {_format_number(total_value)}"


def build_prediction_frame(
    property_input: dict[str, Any],
    current_year: int = 2026,
) -> pd.DataFrame:
    """Convert app-style property input into a one-row model feature DataFrame."""
    validate_property_input(property_input)

    text_flags = []
    if bool(property_input.get("is_lux", False)):
        text_flags.append("lux")
    if bool(property_input.get("is_penthouse", False)):
        text_flags.append("penthouse")
    if bool(property_input.get("is_duplex", False)):
        text_flags.append("duplex")

    raw_row = {
        "title": "Procena nekretnine",
        "description": " ".join(text_flags) or "Korisnički unos za procenu cene",
        "area_m2": float(property_input["area_m2"]),
        "price_eur": np.nan,
        "city": str(property_input.get("city", "Nepoznato")).strip() or "Nepoznato",
        "region": str(property_input.get("region", "Nepoznato")).strip() or "Nepoznato",
        "street": str(property_input.get("street", "Nepoznato")).strip() or "Nepoznato",
        "heating_type": str(property_input.get("heating_type", "Nepoznato")).strip() or "Nepoznato",
        "rooms": _optional_float(property_input.get("rooms")),
        "parking": str(property_input.get("parking", "Nepoznato")).strip() or "Nepoznato",
        "raw_floor_string": build_raw_floor_string(
            property_input.get("floor"),
            property_input.get("total_floors"),
        ),
        "year_built": _optional_float(property_input.get("year_built")),
        "url": "",
    }

    frame = add_model_features(pd.DataFrame([raw_row]), current_year=current_year)
    frame.loc[0, "is_lux"] = int(bool(property_input.get("is_lux", False)))
    frame.loc[0, "is_penthouse"] = int(bool(property_input.get("is_penthouse", False)))
    frame.loc[0, "is_duplex"] = int(bool(property_input.get("is_duplex", False)))
    return frame


def load_prediction_artifact(
    artifact_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
):
    """Load the persisted sklearn pipeline and metadata for prediction."""
    artifact = Path(artifact_path) if artifact_path is not None else DEFAULT_MODELS_DIR / MODEL_ARTIFACT_NAME
    metadata_file = Path(metadata_path) if metadata_path is not None else DEFAULT_MODELS_DIR / MODEL_METADATA_NAME

    if not artifact.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {artifact}. Run: uv run python -m src.training"
        )

    pipeline = joblib.load(artifact)
    metadata = {}
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    return pipeline, metadata


def predict_price(
    pipeline,
    property_input: dict[str, Any],
    current_year: int = 2026,
) -> float:
    """Predict real-estate price in EUR from app-style property input."""
    frame = build_prediction_frame(property_input, current_year=current_year)
    X = frame[get_model_feature_columns()]
    prediction = float(pipeline.predict(X)[0])
    return max(0.0, prediction)


def estimate_price_range(predicted_price: float, mae: float | None) -> tuple[float, float] | None:
    """Return an approximate prediction interval using model MAE for display."""
    if mae is None or pd.isna(mae):
        return None
    lower = max(0.0, predicted_price - float(mae))
    upper = predicted_price + float(mae)
    return lower, upper
