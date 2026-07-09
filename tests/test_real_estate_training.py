import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.model_pipeline import build_model_pipeline
from src.training import (
    DEFAULT_MODEL_CANDIDATES,
    MODEL_ARTIFACT_NAME,
    MODEL_METADATA_NAME,
    MODEL_REGISTRY_ARTIFACT_NAME,
    build_model_metadata,
    collect_ui_options,
    prepare_modeling_data,
    train_candidate_models,
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
            "extra_trees_log_target",
            "hist_gradient_boosting_log_target",
            "catboost",
            "catboost_log_target",
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
        self.assertIn("Vracar", options["regions_by_city"]["Beograd"])
        self.assertNotIn("Grbavica", options["regions_by_city"]["Beograd"])
        self.assertIn("Grbavica", options["regions_by_city"]["Novi Sad"])

    def test_prepare_modeling_data_returns_raw_model_frame_and_cleaning_summary(self):
        raw_df = build_training_fixture()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "fixture.csv"
            raw_df.to_csv(data_path, index=False)

            loaded_raw, model_df, cleaning_summary = prepare_modeling_data(data_path)

        self.assertEqual(len(loaded_raw), len(raw_df))
        self.assertGreater(len(model_df), 0)
        self.assertIn("building_age", model_df.columns)
        self.assertIn("price_eur", model_df.columns)
        self.assertEqual(cleaning_summary["raw_rows"], len(raw_df))
        self.assertEqual(cleaning_summary["cleaned_rows"], len(model_df))

    def test_build_model_metadata_contains_portable_paths_metrics_ui_options_and_counts(self):
        raw_df = build_training_fixture()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "fixture.csv"
            artifact_path = Path(tmpdir) / "models" / MODEL_ARTIFACT_NAME
            artifact_path.parent.mkdir()
            raw_df.to_csv(data_path, index=False)
            _, model_df, cleaning_summary = prepare_modeling_data(data_path)
            training_result = train_candidate_models(
                model_df,
                candidate_specs=[
                    ("Baseline - DummyRegressor median", "dummy_median"),
                    ("LinearRegression", "linear_regression"),
                ],
                test_size=0.25,
                random_state=42,
            )

            metadata = build_model_metadata(
                training_result=training_result,
                model_df=model_df,
                cleaning_summary=cleaning_summary,
                artifact_path=artifact_path,
                data_path=data_path,
                random_state=42,
                test_size=0.25,
            )

        self.assertEqual(metadata["feature_columns"], training_result["feature_columns"])
        self.assertEqual(len(metadata["metrics"]), 2)
        self.assertEqual(len(metadata["available_models"]), 2)
        self.assertIn(metadata["best_model_key"], {"dummy_median", "linear_regression"})
        self.assertEqual(metadata["raw_rows"], len(raw_df))
        self.assertEqual(metadata["cleaned_rows"], len(model_df))
        self.assertEqual(metadata["test_size"], 0.25)
        self.assertEqual(metadata["random_state"], 42)
        self.assertIn("cities", metadata["ui_options"])
        self.assertIn("mae", metadata["metrics"][0])

    def test_train_candidate_models_uses_tqdm_when_progress_is_enabled(self):
        raw_df = build_training_fixture()
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "fixture.csv"
            raw_df.to_csv(data_path, index=False)
            _, model_df, _ = prepare_modeling_data(data_path)

            tqdm_calls = []

            def fake_tqdm(iterable, **kwargs):
                tqdm_calls.append(kwargs)
                return iterable

            with patch("src.training.tqdm", side_effect=fake_tqdm):
                result = train_candidate_models(
                    model_df,
                    candidate_specs=[
                        ("Baseline - DummyRegressor median", "dummy_median"),
                        ("LinearRegression", "linear_regression"),
                    ],
                    test_size=0.25,
                    random_state=42,
                    show_progress=True,
                )

        self.assertEqual(len(tqdm_calls), 1)
        self.assertEqual(tqdm_calls[0]["desc"], "Training model candidates")
        self.assertEqual(tqdm_calls[0]["unit"], "model")
        self.assertFalse(tqdm_calls[0]["disable"])
        self.assertEqual(len(result["metrics_table"]), 2)

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
            registry_path = models_dir / MODEL_REGISTRY_ARTIFACT_NAME
            metadata_path = models_dir / MODEL_METADATA_NAME
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            pipeline = joblib.load(artifact_path)
            registry = joblib.load(registry_path)

            self.assertTrue(artifact_path.exists())
            self.assertTrue(registry_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertEqual(result["artifact_path"], str(artifact_path))
            self.assertEqual(result["model_registry_path"], str(registry_path))
            self.assertEqual(metadata["model_registry_path"], str(registry_path.resolve()))
            self.assertEqual(len(metadata["metrics"]), 3)
            self.assertEqual(len(metadata["available_models"]), 3)
            self.assertEqual(len(registry["pipelines"]), 3)
            self.assertIn(metadata["best_model_key"], {"linear_regression", "random_forest", "gradient_boosting"})
            self.assertGreater(metadata["cleaned_rows"], 0)
            self.assertIn("cities", metadata["ui_options"])
            predictions = pipeline.predict(result["X_test"])
            self.assertTrue(np.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
