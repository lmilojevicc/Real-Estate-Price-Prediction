import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.model_pipeline import build_model_pipeline
from src.training import (
    DEFAULT_MODEL_CANDIDATES,
    MODEL_ARTIFACT_NAME,
    MODEL_METADATA_NAME,
    collect_ui_options,
    train_and_save_best_model,
)


def build_training_fixture() -> pd.DataFrame:
    rows = []
    base_rows = [
        ("Beograd", "Vracar", 42.0, 1.5, 115_000.0, "Centralno", "Da", "1 / 5", 2005.0, "Opis"),
        ("Beograd", "Novi Beograd", 58.0, 2.0, 155_000.0, "Centralno", "Ne", "3 / 8", 2015.0, "Opis"),
        ("Beograd", "Centar", 80.0, 3.0, 260_000.0, "Etažno", "Da", "5 / 5", 2020.0, "Lux stan"),
        ("Novi Sad", "Grbavica", 45.0, 1.5, 105_000.0, "Gas", "Ne", "2 / 4", 2000.0, "Opis"),
        ("Novi Sad", "Centar", 67.0, 2.5, 180_000.0, "Centralno", "Da", "4 / 6", 2018.0, "Opis"),
        ("Novi Sad", "Telep", 92.0, 3.5, 220_000.0, "Etažno", "Da", "1 / 3", 2012.0, "duplex"),
        ("Niš", "Centar", 38.0, 1.0, 72_000.0, "TA peć", "Ne", "Prizemlje", 1995.0, "Opis"),
        ("Niš", "Bulevar", 64.0, 2.5, 118_000.0, "Centralno", "Ne", "6 / 8", 2010.0, "Opis"),
        ("Kragujevac", "Centar", 52.0, 2.0, 83_000.0, "Etažno", "Da", "2 / 5", 2008.0, "Opis"),
        ("Zlatibor", "Centar", 44.0, 1.5, 132_000.0, "Podno", "Da", "3 / 4", 2022.0, "penthouse"),
        ("Pančevo", "Centar", 73.0, 3.0, 126_000.0, "Gas", "Ne", "1 / 4", 2002.0, "Opis"),
        ("Subotica", "Centar", 88.0, 3.5, 140_000.0, "Etažno", "Da", "2 / 2", 1998.0, "Opis"),
    ]
    for index, (city, region, area, rooms, price, heating, parking, floor, year, description) in enumerate(base_rows):
        rows.append(
            {
                "title": f"Stan {index}",
                "description": description,
                "area_m2": area,
                "price_eur": price,
                "city": city,
                "region": region,
                "street": "Nepoznato",
                "heating_type": heating,
                "rooms": rooms,
                "parking": parking,
                "raw_floor_string": floor,
                "year_built": year,
                "url": f"https://example.com/{index}",
            }
        )
    return pd.DataFrame(rows)


class RealEstateTrainingTests(unittest.TestCase):
    def test_model_candidate_registry_includes_cs490_required_and_extra_regressors(self):
        candidate_keys = [model_key for _, model_key in DEFAULT_MODEL_CANDIDATES]

        for required_key in [
            "linear_regression",
            "random_forest",
            "gradient_boosting",
            "extra_trees",
        ]:
            with self.subTest(required_key=required_key):
                self.assertIn(required_key, candidate_keys)
                self.assertIsNotNone(build_model_pipeline(required_key))

    def test_collect_ui_options_uses_dataset_categories_for_streamlit_controls(self):
        options = collect_ui_options(build_training_fixture())

        self.assertIn("Beograd", options["cities"])
        self.assertIn("Novi Sad", options["cities"])
        self.assertIn("Centralno", options["heating_types"])
        self.assertIn("Da", options["parking_options"])

    def test_train_and_save_best_model_writes_prediction_artifact_and_metadata(self):
        raw_df = build_training_fixture()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "fixture.csv"
            models_dir = Path(tmpdir) / "models"
            raw_df.to_csv(data_path, index=False)

            result = train_and_save_best_model(
                data_path=data_path,
                models_dir=models_dir,
                candidate_specs=[
                    ("LinearRegression", "linear_regression"),
                    ("RandomForestRegressor", "random_forest"),
                    ("GradientBoostingRegressor", "gradient_boosting"),
                ],
                test_size=0.25,
                random_state=42,
            )

            artifact_path = models_dir / MODEL_ARTIFACT_NAME
            metadata_path = models_dir / MODEL_METADATA_NAME
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            pipeline = joblib.load(artifact_path)

            self.assertTrue(artifact_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertEqual(result["artifact_path"], str(artifact_path))
            self.assertEqual(len(metadata["metrics"]), 3)
            self.assertIn(metadata["best_model_key"], {"linear_regression", "random_forest", "gradient_boosting"})
            self.assertGreater(metadata["cleaned_rows"], 0)
            self.assertIn("cities", metadata["ui_options"])
            predictions = pipeline.predict(result["X_test"])
            self.assertTrue(np.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
