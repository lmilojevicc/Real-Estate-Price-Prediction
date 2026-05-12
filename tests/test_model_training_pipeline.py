import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.features import MODEL_CATEGORICAL_FEATURES, MODEL_NUMERIC_FEATURES, add_model_features
from src.model_pipeline import build_model_pipeline, build_preprocessor, get_model_feature_columns


class ModelTrainingPipelineTests(unittest.TestCase):
    def test_get_model_feature_columns_returns_numeric_and_categorical_inputs(self):
        columns = get_model_feature_columns()

        for column in MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES:
            with self.subTest(column=column):
                self.assertIn(column, columns)

    def test_preprocessor_imputes_missing_values_and_handles_unseen_categories(self):
        train = pd.DataFrame(
            [
                {"area_m2": 50.0, "rooms": 2.0, "floor": 1.0, "total_floors": 5.0, "is_last_floor": 0.0, "year_built": 2010.0, "building_age": 16.0, "is_lux": 0, "is_penthouse": 0, "is_duplex": 0, "city": "Beograd", "region": "Vracar", "heating_type": "Centralno", "parking": "Da"},
                {"area_m2": 70.0, "rooms": np.nan, "floor": np.nan, "total_floors": np.nan, "is_last_floor": np.nan, "year_built": np.nan, "building_age": np.nan, "is_lux": 1, "is_penthouse": 0, "is_duplex": 0, "city": "Novi Sad", "region": "Grbavica", "heating_type": None, "parking": "Ne"},
            ]
        )
        test = pd.DataFrame(
            [
                {"area_m2": 65.0, "rooms": 2.5, "floor": 3.0, "total_floors": 6.0, "is_last_floor": 0.0, "year_built": 2015.0, "building_age": 11.0, "is_lux": 0, "is_penthouse": 1, "is_duplex": 0, "city": "Kragujevac", "region": "Centar", "heating_type": "Gas", "parking": "Nepoznato"}
            ]
        )

        preprocessor = build_preprocessor()
        transformed_train = preprocessor.fit_transform(train)
        transformed_test = preprocessor.transform(test)

        self.assertEqual(transformed_train.shape[0], 2)
        self.assertEqual(transformed_test.shape[0], 1)
        train_values = transformed_train.toarray() if hasattr(transformed_train, "toarray") else transformed_train
        test_values = transformed_test.toarray() if hasattr(transformed_test, "toarray") else transformed_test
        self.assertFalse(np.isnan(train_values).any())
        self.assertFalse(np.isnan(test_values).any())

    def test_build_model_pipeline_fits_and_predicts_real_estate_prices(self):
        raw = pd.DataFrame(
            [
                {"title": "Stan 1", "description": "Opis", "area_m2": 40.0, "price_eur": 80_000.0, "city": "Beograd", "region": "Centar", "street": "A", "heating_type": "Centralno", "rooms": 1.5, "parking": "Ne", "raw_floor_string": "1/4", "year_built": 2000.0},
                {"title": "Stan 2", "description": "Opis", "area_m2": 55.0, "price_eur": 120_000.0, "city": "Beograd", "region": "Vracar", "street": "B", "heating_type": "Centralno", "rooms": 2.0, "parking": "Da", "raw_floor_string": "2/5", "year_built": 2010.0},
                {"title": "Stan 3", "description": "Lux", "area_m2": 80.0, "price_eur": 220_000.0, "city": "Novi Sad", "region": "Grbavica", "street": "C", "heating_type": "Gas", "rooms": 3.0, "parking": "Da", "raw_floor_string": "5/5", "year_built": 2020.0},
                {"title": "Stan 4", "description": "Opis", "area_m2": 35.0, "price_eur": 70_000.0, "city": "Novi Sad", "region": "Centar", "street": "D", "heating_type": "TA", "rooms": 1.0, "parking": "Ne", "raw_floor_string": "Prizemlje", "year_built": 1990.0},
            ]
        )
        model_df = add_model_features(raw)
        X = model_df[get_model_feature_columns()]
        y = model_df["price_eur"]

        pipeline = build_model_pipeline("linear_regression")
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        self.assertIsInstance(pipeline, Pipeline)
        self.assertEqual(len(predictions), len(y))
        self.assertTrue(np.isfinite(predictions).all())

    def test_build_model_pipeline_supports_required_cs490_regressors(self):
        required_model_keys = [
            "linear_regression",
            "random_forest",
            "gradient_boosting",
        ]

        for model_key in required_model_keys:
            with self.subTest(model_key=model_key):
                pipeline = build_model_pipeline(model_key)
                self.assertIsInstance(pipeline, Pipeline)

    def test_build_model_pipeline_rejects_unknown_model_name(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            build_model_pipeline("unsupported_model")


if __name__ == "__main__":
    unittest.main()
