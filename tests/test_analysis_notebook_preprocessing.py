import json
import unittest
from pathlib import Path

from src.data import load_raw_dataset
from src.data_cleaning import build_cleaned_dataset


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "analysis.ipynb"
DATA_PATH = ROOT / "data" / "nekretnine_raw.csv"


class AnalysisNotebookPreprocessingTests(unittest.TestCase):
    def test_analysis_notebook_documents_cleaning_and_feature_pipeline(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = "\n".join(
            "".join(cell.get("source", "")) for cell in notebook.get("cells", [])
        )

        expected_fragments = [
            "data/nekretnine_raw.csv",
            "from src.data import load_raw_dataset, validate_raw_dataset",
            "from src.data_cleaning import",
            "from src.features import",
            "from src.preprocessing import build_preprocessor, get_model_feature_columns",
            "composite_primary_key",
            "composite_duplicate_rows",
            "rows_removed_if_keeping_first",
            "before_after_summary",
            "outlier_summary",
            "outlier_examples",
            "raw_df",
            "deduped_df",
            "cleaned_df",
            "Distribucije pre i posle ciscenja",
            "Cena pre ciscenja",
            "Cena posle ciscenja",
            "Kvadratura pre ciscenja",
            "Kvadratura posle ciscenja",
            "price_per_m2_by_city",
            "ColumnTransformer",
            "SimpleImputer",
            "OneHotEncoder",
            "train_test_split",
            "model_df = add_model_features(cleaned_df)",
            "preprocessor = build_preprocessor()",
            "X_train_prepared = preprocessor.fit_transform(X_train)",
            "X_test_prepared = preprocessor.transform(X_test)",
            "Nedostajuce vrednosti posle preprocessinga",
            "price_eur nije u ulaznim kolonama",
        ]
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, sources)

    def test_analysis_notebook_cleaning_rules_remove_invalid_model_rows(self):
        df = load_raw_dataset(DATA_PATH)
        cleaned = build_cleaned_dataset(df)

        self.assertGreater(len(df) - len(cleaned), 0)
        self.assertTrue(cleaned["area_m2"].between(10, 500).all())
        self.assertTrue(cleaned["price_eur"].between(10_000, 1_000_000).all())
        self.assertTrue((cleaned["rooms"].isna() | cleaned["rooms"].between(0.5, 10)).all())
        self.assertTrue(
            (cleaned["year_built"].isna() | cleaned["year_built"].between(1800, 2028)).all()
        )


if __name__ == "__main__":
    unittest.main()
