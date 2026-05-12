"""Cleaning utilities for the real-estate listings dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd


COMPOSITE_PRIMARY_KEY = ["title", "description", "area_m2", "price_eur", "city"]


def split_complete_and_incomplete_key_rows(
    df: pd.DataFrame,
    composite_key: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by whether all composite-key fields are present."""
    key = composite_key or COMPOSITE_PRIMARY_KEY
    complete_key_df = df.dropna(subset=key).copy()
    incomplete_key_df = df[df[key].isna().any(axis=1)].copy()
    return complete_key_df, incomplete_key_df


def build_deduped_dataset(
    df: pd.DataFrame,
    composite_key: list[str] | None = None,
) -> pd.DataFrame:
    """Drop duplicate complete composite-key rows while preserving incomplete-key rows."""
    key = composite_key or COMPOSITE_PRIMARY_KEY
    complete_key_df, incomplete_key_df = split_complete_and_incomplete_key_rows(df, key)
    deduped_complete_df = complete_key_df.drop_duplicates(subset=key, keep="first")
    return pd.concat([deduped_complete_df, incomplete_key_df], ignore_index=True)


def build_cleaned_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return model-ready rows after deduplication and basic quality filters."""
    deduped = build_deduped_dataset(df)

    valid_area = deduped["area_m2"].between(10, 500)
    valid_price = deduped["price_eur"].between(10_000, 1_000_000)
    valid_rooms = deduped["rooms"].isna() | deduped["rooms"].between(0.5, 10)
    valid_year = deduped["year_built"].isna() | deduped["year_built"].between(1800, 2028)

    cleaned = deduped[valid_area & valid_price & valid_rooms & valid_year].copy()
    return cleaned.reset_index(drop=True)


def create_cleaning_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize duplicate removal and model-quality filtering counts."""
    complete_key_df, _ = split_complete_and_incomplete_key_rows(df)
    duplicate_rows = len(
        complete_key_df[
            complete_key_df.duplicated(subset=COMPOSITE_PRIMARY_KEY, keep=False)
        ]
    )
    duplicate_groups = complete_key_df.groupby(
        COMPOSITE_PRIMARY_KEY,
        dropna=False,
    ).size()
    rows_removed_if_keeping_first = int(
        complete_key_df.duplicated(subset=COMPOSITE_PRIMARY_KEY, keep="first").sum()
    )
    deduped = build_deduped_dataset(df)
    cleaned = build_cleaned_dataset(df)

    return {
        "raw_rows": len(df),
        "complete_key_rows": len(complete_key_df),
        "duplicate_rows": duplicate_rows,
        "duplicate_groups": int((duplicate_groups > 1).sum()),
        "rows_removed_if_keeping_first": rows_removed_if_keeping_first,
        "deduped_rows": len(deduped),
        "cleaned_rows": len(cleaned),
        "rows_removed_total": len(df) - len(cleaned),
    }
