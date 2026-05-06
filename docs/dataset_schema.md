# Real Estate Dataset Schema

This dataset is specifically designed and engineered for Machine Learning training.

## Part 1: Base Scraped Features (Raw Data)

| Original Field (RS) | CSV Header         | Data Type | ML Handling / Null Logic                                      | Example                          |
| :------------------ | :----------------- | :-------- | :------------------------------------------------------------ | :------------------------------- |
| Naslov oglasa       | `title`            | string    | Raw text. Empty string if missing.                            | "Nov dvosoban stan, Vračar"      |
| Opis oglasa         | `description`      | string    | Raw text. Empty string if missing.                            | "Luksuzan stan u novogradnji..." |
| Kvadratura          | `area_m2`          | float     | Extracted number only. Blank (`NaN`) if missing.              | 45.5                             |
| Cena                | `price_eur`        | float     | Extracted number only. Blank (`NaN`) if missing.              | 120000.0                         |
| Lokacija (City)     | `city`             | string    | Split from location string (index 2). "Nepoznato" if missing. | "Beograd"                        |
| Lokacija (Region)   | `region`           | string    | Split from location string (index 3). "Nepoznato" if missing. | "Vračar"                         |
| Lokacija (Street)   | `street`           | string    | Split from location string (index 4). "Nepoznato" if missing. | "Hram Svetog Save"               |
| Tip Grejanja        | `heating_type`     | string    | Raw text. "Nepoznato" if missing.                             | "Centralno"                      |
| Sobe                | `rooms`            | float     | Extracted number only. Blank (`NaN`) if missing.              | 2.5                              |
| Parking             | `parking`          | string    | Raw text. "Nepoznato" if missing.                             | "Da"                             |
| Spratnost           | `raw_floor_string` | string    | Raw text directly from the website. "Nepoznato" if missing.   | "3/5", "Suteren"                 |
| Godina izgradnje    | `year_built`       | float     | Extracted year. Blank (`NaN`) if missing.                     | 2019.0                           |
| Link                | `url`              | string    | Full absolute URL to listing. Empty string (`""`) if missing. | "https://www.nekretnine.rs/..."  |

## Part 2: Calculated ML Features (Feature Engineering)

These features are generated from the base dataset in the notebook preprocessing pipeline, preparing the data for the Machine Learning models.

| Base Feature Source     | CSV Header      | Data Type | Feature Engineering Logic                                                                                            | Example |
| :---------------------- | :-------------- | :-------- | :------------------------------------------------------------------------------------------------------------------- | :------ |
| `price_eur`, `area_m2`  | `price_per_m2`  | float     | `price_eur` / `area_m2` rounded to 2 decimal places. Blank (`NaN`) if either base value is missing or 0.             | 2637.36 |
| `year_built`            | `building_age`  | float     | Current Year - `year_built`. Blank (`NaN`) if missing.                                                               | 5.0     |
| `title`, `desc`         | `is_lux`        | int (1/0) | `1` if "lux" or "luks" (case-insensitive) in title or description, else `0`.                                         | 1       |
| `title`, `desc`         | `is_penthouse`  | int (1/0) | `1` if "penthouse" or "penthaus" in title or description, else `0`.                                                  | 0       |
| `title`, `desc`         | `is_duplex`     | int (1/0) | `1` if "duplex" or "dupleks" in title or description, else `0`.                                                      | 0       |
| `raw_floor_string`      | `floor`         | float     | Parsed from raw string. Suteren = `-1.0`, Prizemlje/Visoko prizemlje = `0.0`. Missing/unparseable = Blank (`NaN`).   | 3.0     |
| `raw_floor_string`      | `total_floors`  | float     | Parsed from format like "3/5" as float. Blank (`NaN`) if missing.                                                    | 5.0     |
| `floor`, `total_floors` | `is_last_floor` | float     | `1` if `floor` == `total_floors`, `0` if both exist and are not equal, `NaN` if either value is missing/unparseable. | 1       |
