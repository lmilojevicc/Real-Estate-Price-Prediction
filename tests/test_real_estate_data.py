import unittest
import tempfile
from pathlib import Path

import pandas as pd

from src.data import (
    EXPECTED_RAW_COLUMNS,
    RealEstateDataValidationError,
    load_raw_dataset,
    validate_raw_dataset,
)


class RealEstateDataLoadingTests(unittest.TestCase):
    def build_valid_raw_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "title": ["Stan"],
                "description": ["Opis"],
                "area_m2": [55.0],
                "price_eur": [120000.0],
                "city": ["Beograd"],
                "region": ["Vracar"],
                "street": ["Glavna"],
                "heating_type": ["Centralno"],
                "rooms": [2.0],
                "parking": ["Da"],
                "raw_floor_string": ["2 / 5"],
                "year_built": [2010.0],
                "url": ["https://example.test/stan"],
            }
        )

    def test_load_raw_dataset_reads_canonical_file_with_expected_schema(self):
        df = load_raw_dataset()

        self.assertGreaterEqual(len(df), 1000)
        self.assertTrue(set(EXPECTED_RAW_COLUMNS).issubset(df.columns))
        self.assertIn("price_eur", df.columns)
        self.assertIn("area_m2", df.columns)
        self.assertIn("city", df.columns)

    def test_validate_raw_dataset_reports_shape_and_key_columns(self):
        df = self.build_valid_raw_frame()

        summary = validate_raw_dataset(df, min_rows=1)

        self.assertEqual(summary["rows"], 1)
        self.assertEqual(summary["columns"], 13)
        self.assertEqual(summary["target_column"], "price_eur")
        self.assertEqual(summary["numeric_columns"], ["area_m2", "price_eur", "rooms", "year_built"])

    def test_validate_raw_dataset_rejects_missing_columns(self):
        df = pd.DataFrame({"price_eur": [100000.0], "area_m2": [50.0]})

        with self.assertRaisesRegex(RealEstateDataValidationError, "Missing required columns"):
            validate_raw_dataset(df, min_rows=1)

    def test_validate_raw_dataset_rejects_too_few_rows(self):
        df = pd.DataFrame({column: [] for column in EXPECTED_RAW_COLUMNS})

        with self.assertRaisesRegex(RealEstateDataValidationError, "at least 1 rows"):
            validate_raw_dataset(df, min_rows=1)

    def test_validate_raw_dataset_rejects_unparseable_numeric_values(self):
        df = self.build_valid_raw_frame()
        df["area_m2"] = df["area_m2"].astype("object")
        df.loc[0, "area_m2"] = "nije broj"

        with self.assertRaisesRegex(RealEstateDataValidationError, "Numeric columns"):
            validate_raw_dataset(df, min_rows=1)

    def test_load_raw_dataset_can_skip_validation_for_small_fixture(self):
        df = self.build_valid_raw_frame()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "small_fixture.csv"
            df.to_csv(data_path, index=False)

            loaded = load_raw_dataset(data_path, validate=False)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded.loc[0, "city"], "Beograd")

    def test_load_raw_dataset_raises_file_not_found_for_missing_path(self):
        missing_path = Path("does-not-exist") / "nekretnine.csv"

        with self.assertRaisesRegex(FileNotFoundError, "Dataset file does not exist"):
            load_raw_dataset(missing_path)


if __name__ == "__main__":
    unittest.main()
