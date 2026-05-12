import unittest

import pandas as pd

from src.evaluation import compare_regression_results, evaluate_regression_predictions


class ModelEvaluationTests(unittest.TestCase):
    def test_evaluate_regression_predictions_returns_mae_rmse_and_r2(self):
        metrics = evaluate_regression_predictions(
            "Linear Regression",
            y_true=[100_000, 150_000, 200_000],
            y_pred=[110_000, 140_000, 210_000],
        )

        self.assertEqual(metrics["model"], "Linear Regression")
        self.assertAlmostEqual(metrics["mae"], 10_000.0)
        self.assertAlmostEqual(metrics["rmse"], 10_000.0)
        self.assertAlmostEqual(metrics["r2"], 0.94)

    def test_evaluate_regression_predictions_rejects_empty_inputs(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            evaluate_regression_predictions("Empty", y_true=[], y_pred=[])

    def test_compare_regression_results_sorts_models_by_lowest_mae(self):
        results = compare_regression_results(
            [
                {"model": "Random Forest", "mae": 12_000.0, "rmse": 18_000.0, "r2": 0.81},
                {"model": "Linear Regression", "mae": 20_000.0, "rmse": 25_000.0, "r2": 0.70},
                {"model": "Gradient Boosting", "mae": 10_000.0, "rmse": 16_000.0, "r2": 0.85},
            ]
        )

        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(results["model"].tolist(), ["Gradient Boosting", "Random Forest", "Linear Regression"])
        self.assertEqual(results.iloc[0]["rank_by_mae"], 1)
        self.assertEqual(results.iloc[2]["rank_by_mae"], 3)


if __name__ == "__main__":
    unittest.main()
