import unittest

import numpy as np
import pandas as pd

from src.preprocessing import build_preprocessor, get_model_feature_columns
from src import model_pipeline


class RealEstatePreprocessingTests(unittest.TestCase):
    def test_get_model_feature_columns_excludes_target_price(self):
        feature_columns = get_model_feature_columns()

        self.assertIn("area_m2", feature_columns)
        self.assertIn("city", feature_columns)
        self.assertNotIn("price_eur", feature_columns)

    def test_preprocessing_and_model_pipeline_feature_columns_match(self):
        self.assertEqual(get_model_feature_columns(), model_pipeline.get_model_feature_columns())

    def test_build_preprocessor_imputes_and_encodes_model_features(self):
        df = pd.DataFrame(
            [
                {
                    "area_m2": 50.0,
                    "rooms": 2.0,
                    "floor": 1.0,
                    "total_floors": 5.0,
                    "is_last_floor": 0.0,
                    "year_built": 2010.0,
                    "building_age": 16.0,
                    "is_lux": 0,
                    "is_penthouse": 0,
                    "is_duplex": 0,
                    "city": "Beograd",
                    "region": "Vracar",
                    "heating_type": "Centralno",
                    "parking": "Da",
                },
                {
                    "area_m2": 80.0,
                    "rooms": np.nan,
                    "floor": np.nan,
                    "total_floors": np.nan,
                    "is_last_floor": np.nan,
                    "year_built": np.nan,
                    "building_age": np.nan,
                    "is_lux": 1,
                    "is_penthouse": 0,
                    "is_duplex": 1,
                    "city": "Novi Sad",
                    "region": None,
                    "heating_type": None,
                    "parking": "Ne",
                },
            ]
        )

        preprocessor = build_preprocessor()
        transformed = preprocessor.fit_transform(df[get_model_feature_columns()])

        self.assertEqual(transformed.shape[0], 2)
        self.assertGreater(transformed.shape[1], len(get_model_feature_columns()))
        self.assertEqual(int(np.isnan(transformed).sum()), 0)

    def test_preprocessing_and_model_pipeline_transformers_match_expected_features(self):
        preprocessing_transformers = build_preprocessor().transformers
        pipeline_transformers = model_pipeline.build_preprocessor().transformers

        self.assertEqual(
            [(name, columns) for name, _, columns in preprocessing_transformers],
            [(name, columns) for name, _, columns in pipeline_transformers],
        )


if __name__ == "__main__":
    unittest.main()
