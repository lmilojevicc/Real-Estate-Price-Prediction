# Šema skupa podataka za nekretnine

Ovaj dokument opisuje kolone koje se koriste u CS490 projektu za predikciju cena nekretnina. Kanonski fajl je `data/nekretnine_raw.csv`; scraper kod u `scraper/` ostaje samo kao poreklo podataka i ne pokreće se tokom validacije aktivnosti.

## Osnovne scrape kolone

| Polje u oglasu | CSV kolona | Tip | Obrada i nedostajuće vrednosti | Primer |
| :--- | :--- | :--- | :--- | :--- |
| Naslov oglasa | `title` | string | Sirov tekst; prazno ako nedostaje. | "Nov dvosoban stan, Vračar" |
| Opis oglasa | `description` | string | Sirov tekst; prazno ako nedostaje. | "Luksuzan stan u novogradnji..." |
| Kvadratura | `area_m2` | float | Izdvojen numerički deo; `NaN` ako nedostaje. | 45.5 |
| Cena | `price_eur` | float | Izdvojen numerički deo u evrima; `NaN` ako nedostaje. | 120000.0 |
| Grad | `city` | string | Izdvojeno iz lokacije; `Nepoznato` ako nedostaje. | "Beograd" |
| Region/opština | `region` | string | Izdvojeno iz lokacije; `Nepoznato` ako nedostaje. | "Vračar" |
| Ulica | `street` | string | Izdvojeno iz lokacije; `Nepoznato` ako nedostaje. | "Hram Svetog Save" |
| Grejanje | `heating_type` | string | Tekst iz oglasa; `Nepoznato` ako nedostaje. | "Centralno" |
| Broj soba | `rooms` | float | Izdvojen numerički deo; `NaN` ako nedostaje. | 2.5 |
| Parking | `parking` | string | Tekst iz oglasa; `Nepoznato` ako nedostaje. | "Da" |
| Spratnost | `raw_floor_string` | string | Originalni tekst spratnosti; `Nepoznato` ako nedostaje. | "3/5", "Suteren" |
| Godina izgradnje | `year_built` | float | Izdvojena godina; `NaN` ako nedostaje. | 2019.0 |
| Link | `url` | string | Apsolutni URL oglasa; prazno ako nedostaje. | "https://www.nekretnine.rs/..." |

## Izvedeni atributi za analizu i modele

Izvedeni atributi nastaju u reusable kodu i notebook tokovima pre modelovanja.

| Izvor | Kolona | Tip | Logika | Primer |
| :--- | :--- | :--- | :--- | :--- |
| `price_eur`, `area_m2` | `price_per_m2` | float | `price_eur / area_m2`, zaokruženo na dve decimale; koristi se za EDA, ne kao ulaz modela. | 2637.36 |
| `year_built` | `building_age` | float | Trenutna godina minus `year_built`; `NaN` ako godina nedostaje. | 5.0 |
| `title`, `description` | `is_lux` | int | `1` ako naslov ili opis sadrže "lux" ili "luks", inače `0`. | 1 |
| `title`, `description` | `is_penthouse` | int | `1` ako tekst sadrži "penthouse" ili "penthaus", inače `0`. | 0 |
| `title`, `description` | `is_duplex` | int | `1` ako tekst sadrži "duplex" ili "dupleks", inače `0`. | 0 |
| `raw_floor_string` | `floor` | float | Parsirana spratnost; suteren je `-1.0`, prizemlje je `0.0`, a nepoznato ostaje `NaN`. | 3.0 |
| `raw_floor_string` | `total_floors` | float | Ukupan broj spratova parsiran iz formata kao što je `3/5`; `NaN` ako nedostaje. | 5.0 |
| `floor`, `total_floors` | `is_last_floor` | float | `1` ako je stan na poslednjem spratu, `0` ako nije, `NaN` ako podaci nisu dostupni. | 1 |

## Model input contract

Ulazni atributi modela su:

```text
area_m2, rooms, floor, total_floors, is_last_floor, year_built, building_age,
is_lux, is_penthouse, is_duplex, city, region, heating_type, parking
```

Ciljna kolona je `price_eur`. Kolone izvedene direktno iz ciljne promenljive, posebno `price_per_m2`, ne ulaze u model da ne bi došlo do curenja ciljne informacije.
