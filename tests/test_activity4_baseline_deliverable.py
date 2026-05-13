import unittest
from pathlib import Path

import numpy as np

from src.features import add_model_features
from src.model_pipeline import get_model_feature_columns
from src.training import train_candidate_models
from tests.test_real_estate_training import build_training_fixture


ROOT = Path(__file__).resolve().parents[1]


class Activity4BaselineDeliverableTests(unittest.TestCase):
    def test_activity4_first_model_trains_with_reusable_training_code(self):
        model_df = add_model_features(build_training_fixture())

        result = train_candidate_models(
            model_df=model_df,
            candidate_specs=[
                ("Baseline - DummyRegressor median", "dummy_median"),
                ("LinearRegression", "linear_regression"),
            ],
            test_size=0.25,
            random_state=42,
        )

        self.assertEqual(
            set(result["metrics_table"]["model"]),
            {"Baseline - DummyRegressor median", "LinearRegression"},
        )
        self.assertEqual(result["feature_columns"], get_model_feature_columns())
        self.assertNotIn("price_per_m2", result["feature_columns"])
        for metric in ["mae", "rmse", "r2"]:
            with self.subTest(metric=metric):
                self.assertTrue(np.isfinite(result["metrics_table"][metric]).all())


if __name__ == "__main__":
    unittest.main()
