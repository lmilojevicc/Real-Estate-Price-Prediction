import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import add_model_features
from src.model_pipeline import build_model_pipeline, get_model_feature_columns
from src.prediction import (
    PropertyInputValidationError,
    build_prediction_frame,
    build_raw_floor_string,
    estimate_price_range,
    load_model_registry,
    load_prediction_artifact,
    predict_price,
    validate_property_input,
)
from src.training import MODEL_ARTIFACT_NAME, MODEL_METADATA_NAME, MODEL_REGISTRY_ARTIFACT_NAME


VALID_PROPERTY_INPUT = {
    "city": "Beograd",
    "region": "Vracar",
    "area_m2": 64.0,
    "rooms": 2.5,
    "heating_type": "Centralno",
    "parking": "Da",
    "floor": 3.0,
    "total_floors": 6.0,
    "year_built": 2015.0,
    "is_lux": True,
    "is_penthouse": False,
    "is_duplex": False,
}


def build_prediction_training_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"title": "Stan 1", "description": "Opis", "area_m2": 40.0, "price_eur": 80_000.0, "city": "Beograd", "region": "Centar", "street": "A", "heating_type": "Centralno", "rooms": 1.5, "parking": "Ne", "raw_floor_string": "1 / 4", "year_built": 2000.0},
            {"title": "Stan 2", "description": "Opis", "area_m2": 55.0, "price_eur": 120_000.0, "city": "Beograd", "region": "Vracar", "street": "B", "heating_type": "Centralno", "rooms": 2.0, "parking": "Da", "raw_floor_string": "2 / 5", "year_built": 2010.0},
            {"title": "Stan 3", "description": "lux", "area_m2": 80.0, "price_eur": 220_000.0, "city": "Novi Sad", "region": "Grbavica", "street": "C", "heating_type": "Gas", "rooms": 3.0, "parking": "Da", "raw_floor_string": "5 / 5", "year_built": 2020.0},
            {"title": "Stan 4", "description": "Opis", "area_m2": 35.0, "price_eur": 70_000.0, "city": "Novi Sad", "region": "Centar", "street": "D", "heating_type": "TA", "rooms": 1.0, "parking": "Ne", "raw_floor_string": "Prizemlje", "year_built": 1990.0},
            {"title": "Stan 5", "description": "duplex", "area_m2": 100.0, "price_eur": 260_000.0, "city": "Beograd", "region": "Novi Beograd", "street": "E", "heating_type": "Centralno", "rooms": 4.0, "parking": "Da", "raw_floor_string": "8 / 10", "year_built": 2022.0},
        ]
    )


def fit_small_prediction_pipeline():
    model_df = add_model_features(build_prediction_training_data())
    X = model_df[get_model_feature_columns()]
    y = model_df["price_eur"]
    pipeline = build_model_pipeline("linear_regression")
    pipeline.fit(X, y)
    return pipeline


class RealEstatePredictionTests(unittest.TestCase):
    def test_build_prediction_frame_converts_form_input_to_model_features(self):
        frame = build_prediction_frame(VALID_PROPERTY_INPUT, current_year=2026)

        for column in get_model_feature_columns():
            with self.subTest(column=column):
                self.assertIn(column, frame.columns)

        self.assertEqual(frame[get_model_feature_columns()].columns.tolist(), get_model_feature_columns())
        self.assertEqual(frame.loc[0, "city"], "Beograd")
        self.assertEqual(frame.loc[0, "region"], "Vracar")
        self.assertEqual(frame.loc[0, "city_region"], "Beograd | Vracar")
        self.assertEqual(frame.loc[0, "is_lux"], 1)
        self.assertEqual(frame.loc[0, "floor"], 3.0)
        self.assertEqual(frame.loc[0, "total_floors"], 6.0)
        self.assertEqual(frame.loc[0, "building_age"], 11.0)

    def test_validate_property_input_rejects_values_outside_cleaning_ranges(self):
        invalid = dict(VALID_PROPERTY_INPUT, area_m2=5.0, rooms=12.0, year_built=1700.0)

        with self.assertRaises(PropertyInputValidationError) as context:
            validate_property_input(invalid)

        message = str(context.exception)
        self.assertIn("area_m2", message)
        self.assertIn("rooms", message)
        self.assertIn("year_built", message)

    def test_validate_property_input_rejects_blank_required_categories(self):
        invalid = dict(VALID_PROPERTY_INPUT, city=" ", region="", heating_type=None, parking="")

        with self.assertRaises(PropertyInputValidationError) as context:
            validate_property_input(invalid)

        message = str(context.exception)
        self.assertIn("city", message)
        self.assertIn("region", message)
        self.assertIn("heating_type", message)
        self.assertIn("parking", message)

    def test_validate_property_input_rejects_inconsistent_floor_values(self):
        invalid = dict(VALID_PROPERTY_INPUT, floor=9.0, total_floors=4.0)

        with self.assertRaises(PropertyInputValidationError) as context:
            validate_property_input(invalid)

        message = str(context.exception)
        self.assertIn("floor", message)
        self.assertIn("total_floors", message)
        self.assertIn("ne može biti veći", message)

    def test_build_raw_floor_string_formats_basement_ground_numeric_and_missing(self):
        self.assertEqual(build_raw_floor_string(-1, 4), "Suteren / 4")
        self.assertEqual(build_raw_floor_string(0, 3), "Prizemlje / 3")
        self.assertEqual(build_raw_floor_string(2.5, 6), "2.5 / 6")
        self.assertEqual(build_raw_floor_string(None, 6), "Nepoznato")
        self.assertEqual(build_raw_floor_string(5, None), "5")

    def test_predict_price_returns_finite_positive_value_for_valid_input(self):
        pipeline = fit_small_prediction_pipeline()

        prediction = predict_price(pipeline, VALID_PROPERTY_INPUT)

        self.assertIsInstance(prediction, float)
        self.assertTrue(np.isfinite(prediction))
        self.assertGreater(prediction, 0.0)

    def test_predict_price_returns_distinct_values_for_distinct_valid_inputs(self):
        pipeline = fit_small_prediction_pipeline()
        smaller_input = dict(VALID_PROPERTY_INPUT, area_m2=40.0, rooms=1.5, is_lux=False)
        larger_input = dict(
            VALID_PROPERTY_INPUT,
            city="Novi Sad",
            region="Grbavica",
            area_m2=100.0,
            rooms=4.0,
            is_duplex=True,
        )

        smaller_prediction = predict_price(pipeline, smaller_input)
        larger_prediction = predict_price(pipeline, larger_input)

        self.assertTrue(np.isfinite(smaller_prediction))
        self.assertTrue(np.isfinite(larger_prediction))
        self.assertGreater(smaller_prediction, 0.0)
        self.assertGreater(larger_prediction, 0.0)
        self.assertNotAlmostEqual(smaller_prediction, larger_prediction, delta=1.0)

    def test_predict_price_rejects_invalid_property_input_before_inference(self):
        pipeline = fit_small_prediction_pipeline()
        invalid = dict(VALID_PROPERTY_INPUT, floor=8.0, total_floors=3.0)

        with self.assertRaises(PropertyInputValidationError):
            predict_price(pipeline, invalid)

    def test_load_prediction_artifact_reads_pipeline_and_metadata_files(self):
        pipeline = fit_small_prediction_pipeline()
        metadata = {"best_model_name": "LinearRegression", "metrics": []}
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            artifact_path = models_dir / MODEL_ARTIFACT_NAME
            metadata_path = models_dir / MODEL_METADATA_NAME
            joblib.dump(pipeline, artifact_path)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            loaded_pipeline, loaded_metadata = load_prediction_artifact(
                artifact_path=artifact_path,
                metadata_path=metadata_path,
            )

            prediction = predict_price(loaded_pipeline, VALID_PROPERTY_INPUT)
            self.assertEqual(loaded_metadata["best_model_name"], "LinearRegression")
            self.assertGreater(prediction, 0.0)

    def test_load_model_registry_reads_saved_pipelines_and_falls_back_to_best(self):
        pipeline = fit_small_prediction_pipeline()
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            registry_path = models_dir / MODEL_REGISTRY_ARTIFACT_NAME
            joblib.dump({"pipelines": {"LinearRegression": pipeline}}, registry_path)

            registry = load_model_registry(
                default_pipeline=None,
                metadata={"model_registry_path": str(registry_path)},
            )
            fallback_registry = load_model_registry(
                default_pipeline=pipeline,
                metadata={"best_model_name": "LinearRegression"},
                registry_path=models_dir / "missing.joblib",
            )

        self.assertIn("LinearRegression", registry)
        self.assertIn("LinearRegression", fallback_registry)
        self.assertGreater(predict_price(registry["LinearRegression"], VALID_PROPERTY_INPUT), 0.0)

    def test_load_prediction_artifact_raises_clear_error_when_artifact_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / MODEL_ARTIFACT_NAME
            metadata_path = Path(tmpdir) / MODEL_METADATA_NAME

            with self.assertRaisesRegex(FileNotFoundError, "Model artifact not found"):
                load_prediction_artifact(artifact_path=artifact_path, metadata_path=metadata_path)

    def test_estimate_price_range_uses_mae_and_clamps_lower_bound(self):
        self.assertEqual(estimate_price_range(100_000.0, 12_500.0), (87_500.0, 112_500.0))
        self.assertEqual(estimate_price_range(5_000.0, 12_500.0), (0.0, 17_500.0))
        self.assertIsNone(estimate_price_range(100_000.0, None))


if __name__ == "__main__":
    unittest.main()
