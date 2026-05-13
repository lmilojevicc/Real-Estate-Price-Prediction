import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "modeling_activities_04_05.ipynb"


class ModelingNotebookDeliverableTests(unittest.TestCase):
    def test_modeling_notebook_exists_as_valid_jupyter_notebook(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook.get("cells", [])), 10)

    def test_modeling_notebook_separates_activity_4_first_model_from_activity_5_comparison(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = "\n".join(
            "".join(cell.get("source", "")) for cell in notebook.get("cells", [])
        )

        expected_fragments = [
            "# Aktivnosti 4 i 5",
            "## Aktivnost 4 - Prva verzija modela / algoritma",
            "## Aktivnost 5 - Dodatni modeli, evaluacija i status projekta",
            "activity_04_model_key = \"linear_regression\"",
            "activity_05_model_specs",
            "najmanje tri modela",
            "Linear Regression, Random Forest i Gradient Boosting",
            "from src.data_cleaning import build_cleaned_dataset",
            "from src.features import add_model_features",
            "from src.model_pipeline import build_model_pipeline",
            "from src.evaluation import compare_regression_results",
            "DummyRegressor",
            "LinearRegression",
            "\"RandomForestRegressor\"",
            "\"GradientBoostingRegressor\"",
            "model_results_activity_04",
            "model_results_activity_05",
            "mean_absolute_error",
            "root_mean_squared_error",
            "r2_score",
            "actual_vs_predicted",
            "best_model_name",
            "trenutni status projekta",
        ]
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, sources)

    def test_modeling_notebook_counts_only_real_project_models_toward_three_model_requirement(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = "\n".join(
            "".join(cell.get("source", "")) for cell in notebook.get("cells", [])
        )

        required_model_specs = [
            '("LinearRegression", "linear_regression")',
            '("RandomForestRegressor", "random_forest")',
            '("GradientBoostingRegressor", "gradient_boosting")',
        ]
        for model_spec in required_model_specs:
            with self.subTest(model_spec=model_spec):
                self.assertIn(model_spec, sources)

        self.assertIn('baseline_model_key = "dummy_median"', sources)
        self.assertIn("Baseline se ne računa", sources)


if __name__ == "__main__":
    unittest.main()
