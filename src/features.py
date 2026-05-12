"""Feature engineering utilities for real-estate price models."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


MODEL_NUMERIC_FEATURES = [
    "area_m2",
    "rooms",
    "floor",
    "total_floors",
    "is_last_floor",
    "year_built",
    "building_age",
    "is_lux",
    "is_penthouse",
    "is_duplex",
]

MODEL_CATEGORICAL_FEATURES = ["city", "region", "heating_type", "parking"]

TEXT_FEATURE_COLUMNS = ["is_lux", "is_penthouse", "is_duplex"]
CATEGORICAL_COLUMNS_TO_NORMALIZE = [
    "city",
    "region",
    "street",
    "heating_type",
    "parking",
    "raw_floor_string",
]


def extract_total_floors(raw_text: object) -> float:
    """Extract total floor count from strings such as ``3/5`` or return NaN."""
    if pd.isna(raw_text):
        return np.nan
    totals = re.findall(r"/\s*(-?\d+(?:[\.,]\d+)?)", str(raw_text))
    if totals:
        return float(totals[0].replace(",", "."))
    return np.nan


def parse_floor_values(raw_floor_string: object) -> tuple[float, float]:
    """Parse current floor and total floors from Serbian listing floor strings."""
    raw = "" if pd.isna(raw_floor_string) else str(raw_floor_string).strip().lower()
    if not raw or raw in {"nepoznato", "-"}:
        return np.nan, np.nan

    total = extract_total_floors(raw)
    if "suteren" in raw:
        return -1.0, total

    if "visoko prizemlje" in raw or "prizemlje" in raw:
        return 0.0, total

    match = re.search(r"(-?\d+(?:[\.,]\d+)?)\s*/\s*(-?\d+(?:[\.,]\d+)?)", raw)
    if match:
        floor = float(match.group(1).replace(",", "."))
        total_floors = float(match.group(2).replace(",", "."))
        return floor, total_floors

    number_match = re.search(r"^-?\d+(?:[\.,]\d+)?$", raw)
    if number_match:
        return float(raw.replace(",", ".")), np.nan

    return np.nan, np.nan


def normalize_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace empty categorical placeholders with ``Nepoznato``."""
    normalized = df.copy()
    for column in CATEGORICAL_COLUMNS_TO_NORMALIZE:
        if column in normalized.columns:
            series = normalized[column].astype("object")
            normalized[column] = series.mask(series.isin(["-", ""]), np.nan).fillna(
                "Nepoznato"
            )
    return normalized


def add_model_features(df: pd.DataFrame, current_year: int = 2026) -> pd.DataFrame:
    """Add reusable ML features used by notebooks, training code, and Streamlit."""
    model_df = normalize_categorical_values(df)

    model_df["price_per_m2"] = np.where(
        model_df["area_m2"].notna() & (model_df["area_m2"] != 0) & model_df["price_eur"].notna(),
        (model_df["price_eur"] / model_df["area_m2"]).round(2),
        np.nan,
    )

    floor_values = model_df["raw_floor_string"].apply(parse_floor_values)
    model_df["floor"] = [value[0] for value in floor_values]
    model_df["total_floors"] = [value[1] for value in floor_values]
    model_df["is_last_floor"] = np.where(
        model_df["floor"].notna() & model_df["total_floors"].notna(),
        (model_df["floor"] == model_df["total_floors"]).astype(float),
        np.nan,
    )

    model_df["building_age"] = np.where(
        model_df["year_built"].notna(),
        current_year - model_df["year_built"],
        np.nan,
    )

    text = (
        model_df["title"].fillna("") + " " + model_df["description"].fillna("")
    ).str.lower()
    model_df["is_lux"] = text.str.contains(r"\b(?:lux|luks)\b", regex=True).astype(int)
    model_df["is_penthouse"] = text.str.contains(
        r"\b(?:penthouse|penthaus)\b",
        regex=True,
    ).astype(int)
    model_df["is_duplex"] = text.str.contains(
        r"\b(?:duplex|dupleks)\b",
        regex=True,
    ).astype(int)

    return model_df
