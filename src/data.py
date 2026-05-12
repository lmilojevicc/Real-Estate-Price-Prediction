"""Data loading and schema validation for the real-estate dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "nekretnine_raw.csv"

EXPECTED_RAW_COLUMNS = [
    "title",
    "description",
    "area_m2",
    "price_eur",
    "city",
    "region",
    "street",
    "heating_type",
    "rooms",
    "parking",
    "raw_floor_string",
    "year_built",
    "url",
]

REQUIRED_MODELING_COLUMNS = [
    "area_m2",
    "price_eur",
    "city",
    "region",
    "heating_type",
    "rooms",
    "parking",
    "raw_floor_string",
    "year_built",
]

NUMERIC_RAW_COLUMNS = ["area_m2", "price_eur", "rooms", "year_built"]


class RealEstateDataValidationError(ValueError):
    """Raised when the canonical real-estate dataset does not match expectations."""


def _validate_numeric_columns(df: pd.DataFrame) -> None:
    """Ensure numeric source columns are parseable as numbers when present."""
    invalid_columns: list[str] = []
    for column in NUMERIC_RAW_COLUMNS:
        series = df[column]
        converted = pd.to_numeric(series, errors="coerce")
        invalid_mask = converted.isna() & series.notna() & (series.astype(str).str.strip() != "")
        if invalid_mask.any():
            invalid_columns.append(column)

    if invalid_columns:
        joined = ", ".join(invalid_columns)
        raise RealEstateDataValidationError(
            f"Numeric columns contain unparseable values: {joined}"
        )


def validate_raw_dataset(
    df: pd.DataFrame,
    min_rows: int = 1000,
    required_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Validate raw listing data and return a compact schema/shape summary."""
    expected_columns = required_columns or EXPECTED_RAW_COLUMNS
    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise RealEstateDataValidationError(f"Missing required columns: {joined}")

    if len(df) < min_rows:
        raise RealEstateDataValidationError(
            f"Dataset must contain at least {min_rows} rows; found {len(df)}"
        )

    _validate_numeric_columns(df)

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "target_column": "price_eur",
        "expected_columns": list(expected_columns),
        "modeling_columns": list(REQUIRED_MODELING_COLUMNS),
        "numeric_columns": list(NUMERIC_RAW_COLUMNS),
    }


def load_raw_dataset(
    data_path: str | Path = DEFAULT_DATA_PATH,
    min_rows: int = 1000,
    validate: bool = True,
) -> pd.DataFrame:
    """Load the prepared CSV dataset from disk without invoking the scraper."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    df = pd.read_csv(path)
    if validate:
        validate_raw_dataset(df, min_rows=min_rows)
    return df
