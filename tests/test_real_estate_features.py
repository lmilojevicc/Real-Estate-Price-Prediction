import math
import unittest

import pandas as pd

from src.features import (
    add_model_features,
    extract_total_floors,
    normalize_categorical_values,
    parse_floor_values,
)


class RealEstateFeatureEngineeringTests(unittest.TestCase):
    def test_extract_total_floors_reads_number_after_slash(self):
        self.assertEqual(extract_total_floors("3/5"), 5.0)
        self.assertEqual(extract_total_floors("Suteren / 4"), 4.0)
        self.assertTrue(math.isnan(extract_total_floors("Prizemlje")))

    def test_parse_floor_values_handles_numeric_ground_and_basement_labels(self):
        self.assertEqual(parse_floor_values("3/5"), (3.0, 5.0))
        self.assertEqual(parse_floor_values("5 / 5"), (5.0, 5.0))
        self.assertEqual(parse_floor_values("Suteren / 4"), (-1.0, 4.0))
        self.assertEqual(parse_floor_values("Visoko prizemlje / 3"), (0.0, 3.0))
        floor, total = parse_floor_values("Prizemlje")
        self.assertEqual(floor, 0.0)
        self.assertTrue(math.isnan(total))
        floor, total = parse_floor_values("Nepoznato")
        self.assertTrue(math.isnan(floor))
        self.assertTrue(math.isnan(total))

    def test_parse_floor_values_handles_decimal_comma_and_numeric_only_floor(self):
        self.assertEqual(parse_floor_values("2,5 / 6"), (2.5, 6.0))
        floor, total = parse_floor_values("7")

        self.assertEqual(floor, 7.0)
        self.assertTrue(math.isnan(total))

    def test_normalize_categorical_values_replaces_empty_dash_and_none(self):
        df = pd.DataFrame(
            {
                "city": ["", "Beograd"],
                "region": ["-", None],
                "street": [None, "Glavna"],
                "heating_type": ["Centralno", ""],
                "parking": ["-", "Da"],
                "raw_floor_string": ["", "2 / 5"],
                "other": ["-", ""],
            }
        )

        normalized = normalize_categorical_values(df)

        self.assertEqual(normalized.loc[0, "city"], "Nepoznato")
        self.assertEqual(normalized.loc[0, "region"], "Nepoznato")
        self.assertEqual(normalized.loc[0, "street"], "Nepoznato")
        self.assertEqual(normalized.loc[1, "heating_type"], "Nepoznato")
        self.assertEqual(normalized.loc[0, "parking"], "Nepoznato")
        self.assertEqual(normalized.loc[0, "raw_floor_string"], "Nepoznato")
        self.assertEqual(normalized.loc[0, "other"], "-")

    def test_add_model_features_creates_numeric_location_and_text_features(self):
        df = pd.DataFrame(
            [
                {
                    "title": "Lux penthouse dupleks",
                    "description": "Luks stan na poslednjem spratu",
                    "area_m2": 50.0,
                    "price_eur": 125_000.0,
                    "city": "Beograd",
                    "region": "Vracar",
                    "street": "-",
                    "heating_type": "-",
                    "parking": "",
                    "raw_floor_string": "5 / 5",
                    "rooms": 2.0,
                    "year_built": 2016.0,
                },
                {
                    "title": "Porodican stan",
                    "description": None,
                    "area_m2": 80.0,
                    "price_eur": 160_000.0,
                    "city": "Novi Sad",
                    "region": "Grbavica",
                    "street": None,
                    "heating_type": "Centralno",
                    "parking": "Da",
                    "raw_floor_string": "Prizemlje",
                    "rooms": 3.0,
                    "year_built": None,
                },
            ]
        )

        engineered = add_model_features(df, current_year=2026)

        self.assertEqual(engineered.loc[0, "price_per_m2"], 2500.0)
        self.assertEqual(engineered.loc[0, "building_age"], 10.0)
        self.assertEqual(engineered.loc[0, "floor"], 5.0)
        self.assertEqual(engineered.loc[0, "total_floors"], 5.0)
        self.assertEqual(engineered.loc[0, "is_last_floor"], 1.0)
        self.assertEqual(engineered.loc[0, "is_lux"], 1)
        self.assertEqual(engineered.loc[0, "is_penthouse"], 1)
        self.assertEqual(engineered.loc[0, "is_duplex"], 1)
        self.assertEqual(engineered.loc[0, "street"], "Nepoznato")
        self.assertEqual(engineered.loc[0, "heating_type"], "Nepoznato")
        self.assertEqual(engineered.loc[0, "parking"], "Nepoznato")

        self.assertEqual(engineered.loc[1, "price_per_m2"], 2000.0)
        self.assertTrue(pd.isna(engineered.loc[1, "building_age"]))
        self.assertEqual(engineered.loc[1, "floor"], 0.0)
        self.assertTrue(pd.isna(engineered.loc[1, "total_floors"]))
        self.assertTrue(pd.isna(engineered.loc[1, "is_last_floor"]))
        self.assertEqual(engineered.loc[1, "is_lux"], 0)
        self.assertEqual(engineered.loc[1, "is_penthouse"], 0)
        self.assertEqual(engineered.loc[1, "is_duplex"], 0)
        self.assertEqual(engineered.loc[1, "street"], "Nepoznato")

    def test_add_model_features_leaves_source_dataframe_unchanged(self):
        df = pd.DataFrame(
            [
                {
                    "title": "Stan",
                    "description": "Opis",
                    "area_m2": 40.0,
                    "price_eur": 80_000.0,
                    "city": "Beograd",
                    "region": "Centar",
                    "street": "Glavna",
                    "heating_type": "Centralno",
                    "parking": "Ne",
                    "raw_floor_string": "2/4",
                    "rooms": 1.5,
                    "year_built": 2000.0,
                }
            ]
        )

        add_model_features(df)

        self.assertNotIn("price_per_m2", df.columns)
        self.assertNotIn("building_age", df.columns)
        self.assertNotIn("floor", df.columns)

    def test_add_model_features_handles_zero_area_without_infinite_price_per_m2(self):
        df = pd.DataFrame(
            [
                {
                    "title": "Stan",
                    "description": "Opis",
                    "area_m2": 0.0,
                    "price_eur": 80_000.0,
                    "city": "Beograd",
                    "region": "Centar",
                    "street": "Glavna",
                    "heating_type": "Centralno",
                    "parking": "Ne",
                    "raw_floor_string": "2/4",
                    "rooms": 1.5,
                    "year_built": 2000.0,
                }
            ]
        )

        engineered = add_model_features(df)

        self.assertTrue(pd.isna(engineered.loc[0, "price_per_m2"]))


if __name__ == "__main__":
    unittest.main()
