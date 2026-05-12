import unittest

import pandas as pd

from src.data_cleaning import (
    COMPOSITE_PRIMARY_KEY,
    build_cleaned_dataset,
    create_cleaning_summary,
)


class RealEstateCleaningTests(unittest.TestCase):
    def test_build_cleaned_dataset_deduplicates_complete_composite_key_rows(self):
        df = pd.DataFrame(
            [
                {
                    "title": "Stan A",
                    "description": "Isti oglas",
                    "area_m2": 50.0,
                    "price_eur": 100_000.0,
                    "city": "Beograd",
                    "rooms": 2.0,
                    "year_built": 2010.0,
                },
                {
                    "title": "Stan A",
                    "description": "Isti oglas",
                    "area_m2": 50.0,
                    "price_eur": 100_000.0,
                    "city": "Beograd",
                    "rooms": 2.0,
                    "year_built": 2010.0,
                },
                {
                    "title": "Stan B",
                    "description": "Drugi oglas",
                    "area_m2": 65.0,
                    "price_eur": 130_000.0,
                    "city": "Novi Sad",
                    "rooms": 3.0,
                    "year_built": 2005.0,
                },
            ]
        )

        cleaned = build_cleaned_dataset(df)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[COMPOSITE_PRIMARY_KEY].drop_duplicates().shape[0], 2)

    def test_build_cleaned_dataset_keeps_incomplete_key_rows_before_quality_filter(self):
        df = pd.DataFrame(
            [
                {
                    "title": None,
                    "description": "Nedostaje naslov",
                    "area_m2": 42.0,
                    "price_eur": 84_000.0,
                    "city": "Beograd",
                    "rooms": 1.5,
                    "year_built": None,
                },
                {
                    "title": "Nevalidna cena",
                    "description": "Preniska cena",
                    "area_m2": 42.0,
                    "price_eur": 1_000.0,
                    "city": "Beograd",
                    "rooms": 1.5,
                    "year_built": None,
                },
            ]
        )

        cleaned = build_cleaned_dataset(df)

        self.assertEqual(len(cleaned), 1)
        self.assertTrue(cleaned.iloc[0]["title"] is None or pd.isna(cleaned.iloc[0]["title"]))

    def test_build_cleaned_dataset_filters_rows_outside_model_ranges(self):
        df = pd.DataFrame(
            [
                {"title": "valid", "description": "ok", "area_m2": 55.0, "price_eur": 110_000.0, "city": "Beograd", "rooms": 2.0, "year_built": 2020.0},
                {"title": "tiny", "description": "bad", "area_m2": 5.0, "price_eur": 110_000.0, "city": "Beograd", "rooms": 2.0, "year_built": 2020.0},
                {"title": "cheap", "description": "bad", "area_m2": 55.0, "price_eur": 5_000.0, "city": "Beograd", "rooms": 2.0, "year_built": 2020.0},
                {"title": "rooms", "description": "bad", "area_m2": 55.0, "price_eur": 110_000.0, "city": "Beograd", "rooms": 20.0, "year_built": 2020.0},
                {"title": "year", "description": "bad", "area_m2": 55.0, "price_eur": 110_000.0, "city": "Beograd", "rooms": 2.0, "year_built": 1700.0},
            ]
        )

        cleaned = build_cleaned_dataset(df)

        self.assertEqual(cleaned["title"].tolist(), ["valid"])

    def test_create_cleaning_summary_reports_each_cleaning_stage(self):
        df = pd.DataFrame(
            [
                {"title": "valid", "description": "ok", "area_m2": 55.0, "price_eur": 110_000.0, "city": "Beograd", "rooms": 2.0, "year_built": 2020.0},
                {"title": "valid", "description": "ok", "area_m2": 55.0, "price_eur": 110_000.0, "city": "Beograd", "rooms": 2.0, "year_built": 2020.0},
                {"title": "cheap", "description": "bad", "area_m2": 55.0, "price_eur": 5_000.0, "city": "Beograd", "rooms": 2.0, "year_built": 2020.0},
            ]
        )

        summary = create_cleaning_summary(df)

        self.assertEqual(summary["raw_rows"], 3)
        self.assertEqual(summary["complete_key_rows"], 3)
        self.assertEqual(summary["duplicate_rows"], 2)
        self.assertEqual(summary["rows_removed_if_keeping_first"], 1)
        self.assertEqual(summary["deduped_rows"], 2)
        self.assertEqual(summary["cleaned_rows"], 1)


if __name__ == "__main__":
    unittest.main()
