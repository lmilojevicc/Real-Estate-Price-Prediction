"""Streamlit dashboard for CS490 real-estate price prediction."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import create_cleaning_summary
from src.prediction import (
    PropertyInputValidationError,
    estimate_price_range,
    load_prediction_artifact,
    predict_price,
)
from src.training import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODELS_DIR,
    MODEL_ARTIFACT_NAME,
    MODEL_METADATA_NAME,
    prepare_modeling_data,
)


st.set_page_config(
    page_title="Procena cena nekretnina",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def cached_prediction_artifact():
    """Load saved model artifact once per Streamlit session."""
    return load_prediction_artifact(
        artifact_path=DEFAULT_MODELS_DIR / MODEL_ARTIFACT_NAME,
        metadata_path=DEFAULT_MODELS_DIR / MODEL_METADATA_NAME,
    )


@st.cache_data(show_spinner=False)
def cached_modeling_data():
    """Load cleaned/model-ready data for dashboard charts."""
    raw_df, model_df, cleaning_summary = prepare_modeling_data(DEFAULT_DATA_PATH)
    return raw_df, model_df, cleaning_summary


def format_eur(value: float) -> str:
    """Format EUR values with Serbian-style thousands separator."""
    return f"{value:,.0f} EUR".replace(",", ".")


def get_best_model_mae(metadata: dict) -> float | None:
    """Read best-model MAE from saved model metadata."""
    best_model_name = metadata.get("best_model_name")
    for row in metadata.get("metrics", []):
        if row.get("model") == best_model_name:
            return float(row["mae"])
    return None


def safe_options(metadata: dict, key: str, fallback: list[str]) -> list[str]:
    """Return non-empty option lists for Streamlit widgets."""
    options = metadata.get("ui_options", {}).get(key, [])
    if options:
        return options
    return fallback


def render_prediction_tab(metadata: dict, pipeline) -> None:
    """Render the price prediction form and prediction output."""
    st.subheader("Procena cene")
    st.write(
        "Unesite karakteristike nekretnine. Model vraća procenu oglašene cene u evrima."
    )

    city_options = safe_options(metadata, "cities", ["Beograd", "Novi Sad", "Niš"])
    region_options = safe_options(metadata, "regions", ["Centar", "Vračar", "Nepoznato"])
    heating_options = safe_options(metadata, "heating_types", ["Centralno", "Etažno", "Nepoznato"])
    parking_options = safe_options(metadata, "parking_options", ["Da", "Ne", "Nepoznato"])

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            city = st.selectbox("Grad", city_options)
            region = st.selectbox("Region / opština", region_options)
            area_m2 = st.number_input("Kvadratura (m²)", min_value=10.0, max_value=500.0, value=60.0, step=1.0)
            rooms = st.number_input("Broj soba", min_value=0.5, max_value=10.0, value=2.5, step=0.5)
        with col2:
            heating_type = st.selectbox("Grejanje", heating_options)
            parking = st.selectbox("Parking", parking_options)
            floor = st.number_input("Sprat", min_value=-1.0, max_value=100.0, value=2.0, step=1.0)
            total_floors = st.number_input("Ukupno spratova", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
        with col3:
            year_unknown = st.checkbox("Godina izgradnje nepoznata", value=False)
            year_built = None
            if not year_unknown:
                year_built = st.number_input("Godina izgradnje", min_value=1800, max_value=2028, value=2015, step=1)
            is_lux = st.checkbox("Lux oglas")
            is_penthouse = st.checkbox("Penthouse")
            is_duplex = st.checkbox("Duplex")

        submitted = st.form_submit_button("Proceni cenu")

    if not submitted:
        st.info("Popunite formu i kliknite na dugme za procenu cene.")
        return

    property_input = {
        "city": city,
        "region": region,
        "area_m2": area_m2,
        "rooms": rooms,
        "heating_type": heating_type,
        "parking": parking,
        "floor": floor,
        "total_floors": total_floors,
        "year_built": year_built,
        "is_lux": is_lux,
        "is_penthouse": is_penthouse,
        "is_duplex": is_duplex,
    }

    try:
        predicted_price = predict_price(pipeline, property_input)
    except PropertyInputValidationError as exc:
        st.error(f"Nevalidan unos: {exc}")
        return

    mae = get_best_model_mae(metadata)
    estimated_range = estimate_price_range(predicted_price, mae)

    st.metric("Procenjena cena", format_eur(predicted_price))
    if estimated_range is not None:
        lower, upper = estimated_range
        st.caption(
            f"Okvirni raspon na osnovu MAE metrike najboljeg modela: {format_eur(lower)} - {format_eur(upper)}"
        )
    st.caption(
        "Procena je informativna i zavisi od kvaliteta podataka, lokacije i karakteristika koje postoje u datasetu."
    )


def render_data_tab() -> None:
    """Render cleaned dataset context and exploratory charts."""
    st.subheader("Podaci")
    with st.spinner("Učitavanje i priprema podataka..."):
        raw_df, model_df, cleaning_summary = cached_modeling_data()

    col1, col2, col3 = st.columns(3)
    col1.metric("Raw oglasi", f"{len(raw_df):,}".replace(",", "."))
    col2.metric("Model-ready oglasi", f"{len(model_df):,}".replace(",", "."))
    col3.metric("Uklonjeno", f"{cleaning_summary['rows_removed_total']:,}".replace(",", "."))

    st.write("Sažetak čišćenja podataka")
    st.dataframe(pd.DataFrame([cleaning_summary]).T.rename(columns={0: "vrednost"}), use_container_width=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.write("Top gradovi")
        st.bar_chart(model_df["city"].value_counts().head(15))
    with chart_col2:
        st.write("Top regioni")
        st.bar_chart(model_df["region"].value_counts().head(15))

    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        st.write("Raspodela cena")
        st.bar_chart(model_df["price_eur"].dropna() / 1000)
    with dist_col2:
        st.write("Raspodela kvadrature")
        st.bar_chart(model_df["area_m2"].dropna())


def render_models_tab(metadata: dict) -> None:
    """Render model-comparison metadata from model_metadata.json."""
    st.subheader("Modeli")
    st.write("Modeli su trenirani komandom `uv run python -m src.training`.")
    st.caption("Metadata fajl: `models/model_metadata.json`")

    metrics = metadata.get("metrics", [])
    if not metrics:
        st.warning("Model metadata nije dostupna. Pokrenite: `uv run python -m src.training`")
        return

    metrics_df = pd.DataFrame(metrics).sort_values("mae")
    best_model_name = metadata.get("best_model_name", metrics_df.iloc[0]["model"])

    st.metric("Najbolji model prema MAE", best_model_name)
    st.dataframe(metrics_df.round(3), use_container_width=True)

    chart_col1, chart_col2, chart_col3 = st.columns(3)
    with chart_col1:
        st.write("MAE")
        st.bar_chart(metrics_df.set_index("model")["mae"])
    with chart_col2:
        st.write("RMSE")
        st.bar_chart(metrics_df.set_index("model")["rmse"])
    with chart_col3:
        st.write("R²")
        st.bar_chart(metrics_df.set_index("model")["r2"])

    st.write(
        "Najbolji model se bira po najnižoj MAE vrednosti, jer MAE direktno pokazuje prosečnu apsolutnu grešku u evrima."
    )


def render_project_tab(metadata: dict) -> None:
    """Render CS490 project description and architecture summary."""
    st.subheader("O projektu")
    st.markdown(
        """
        **Tema:** Sistem za predikciju cena nekretnina primenom metoda mašinskog učenja.

        **Cilj:** razvoj sistema za analizu tržišta nekretnina i procenu cene na osnovu karakteristika oglasa.

        **Izvor podataka:** oglasi prikupljeni custom web scraper-om sa portala `nekretnine.rs`.

        **Arhitektura:**
        1. `scraper/` prikuplja podatke.
        2. `src.data_cleaning` uklanja duplikate i nevalidne vrednosti.
        3. `src.features` pravi izvedene atribute.
        4. `src.model_pipeline` definiše sklearn preprocessing i modele.
        5. `src.training` trenira modele i čuva najbolji artifact.
        6. `src.prediction` učitava artifact i generiše procenu.
        7. `app/streamlit_app.py` prikazuje dashboard.
        """
    )
    st.info("Ako model nije generisan, prvo pokrenite: `uv run python -m src.training`, zatim `uv run streamlit run app/streamlit_app.py`.")
    if metadata:
        st.json(
            {
                "best_model_name": metadata.get("best_model_name"),
                "raw_rows": metadata.get("raw_rows"),
                "cleaned_rows": metadata.get("cleaned_rows"),
                "train_rows": metadata.get("train_rows"),
                "test_rows": metadata.get("test_rows"),
            }
        )


def main() -> None:
    """Render the Streamlit dashboard."""
    st.title("Procena cena nekretnina")
    st.write("CS490 dashboard za predikciju cena nekretnina u Srbiji.")

    try:
        pipeline, metadata = cached_prediction_artifact()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("uv run python -m src.training", language="bash")
        st.stop()

    prediction_tab, data_tab, models_tab, project_tab = st.tabs(
        ["Procena cene", "Podaci", "Modeli", "O projektu"]
    )

    with prediction_tab:
        render_prediction_tab(metadata, pipeline)
    with data_tab:
        render_data_tab()
    with models_tab:
        render_models_tab(metadata)
    with project_tab:
        render_project_tab(metadata)


if __name__ == "__main__":
    main()
