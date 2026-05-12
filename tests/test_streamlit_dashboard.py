import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "streamlit_app.py"


class StreamlitDashboardTests(unittest.TestCase):
    def test_streamlit_dashboard_file_contains_required_cs490_sections(self):
        source = APP_PATH.read_text(encoding="utf-8")

        expected_fragments = [
            "Procena cena nekretnina",
            "Procena cene",
            "Podaci",
            "Modeli",
            "O projektu",
            "load_prediction_artifact",
            "predict_price",
            "prepare_modeling_data",
            "model_metadata.json",
            "uv run python -m src.training",
        ]
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
