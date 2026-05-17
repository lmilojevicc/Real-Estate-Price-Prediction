"""Streamlit dashboard for CS490 real-estate price prediction."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_cleaning import build_cleaned_dataset, create_cleaning_summary
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
sns.set_theme(style="whitegrid")


@st.cache_resource(show_spinner=False)
def cached_prediction_artifact():
    """Load saved model artifact once per Streamlit session."""
    return load_prediction_artifact(
        artifact_path=DEFAULT_MODELS_DIR / MODEL_ARTIFACT_NAME,
        metadata_path=DEFAULT_MODELS_DIR / MODEL_METADATA_NAME,
    )


@st.cache_data(show_spinner=False)
def cached_modeling_data():
    """Load the same raw, cleaned, and feature-engineered data used by analysis.ipynb."""
    raw_df, model_df, cleaning_summary = prepare_modeling_data(DEFAULT_DATA_PATH)
    cleaned_df = build_cleaned_dataset(raw_df)
    return raw_df, cleaned_df, model_df, cleaning_summary


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


def region_options_for_city(metadata: dict, city: str) -> list[str]:
    """Return only regions/opštine observed for the selected city."""
    fallback = ["Nepoznato"]
    ui_options = metadata.get("ui_options", {})
    regions_by_city = ui_options.get("regions_by_city", {})
    city_regions = regions_by_city.get(city, [])
    if city_regions:
        return city_regions
    if regions_by_city:
        return fallback
    return safe_options(metadata, "regions", fallback)


def render_prediction_tab(metadata: dict, pipeline) -> None:
    """Render the price prediction form and prediction output."""
    st.subheader("Procena cene")
    st.write(
        "Unesite karakteristike nekretnine. Model vraća procenu oglašene cene u evrima."
    )

    city_options = safe_options(metadata, "cities", ["Beograd", "Novi Sad", "Niš"])
    heating_options = safe_options(metadata, "heating_types", ["Centralno", "Etažno", "Nepoznato"])
    parking_options = safe_options(metadata, "parking_options", ["Da", "Ne", "Nepoznato"])

    location_col1, location_col2 = st.columns(2)
    with location_col1:
        city = st.selectbox("Grad", city_options, key="prediction_city")
    region_options = region_options_for_city(metadata, city)
    region_state_key = "prediction_region"
    if st.session_state.get(region_state_key) not in region_options:
        st.session_state[region_state_key] = region_options[0]
    with location_col2:
        region = st.selectbox(
            "Region / opština",
            region_options,
            key=region_state_key,
        )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
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
        "Procena je orijentaciona i zavisi od kvaliteta podataka, lokacije i karakteristika koje postoje u datasetu."
    )


def render_matplotlib_figure(fig) -> None:
    """Render a Matplotlib figure in Streamlit and close it afterwards."""
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def render_distribution_charts(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> None:
    """Recreate the analysis notebook's before/after cleaning histograms."""
    price_upper = raw_df["price_eur"].quantile(0.99)
    area_upper = raw_df["area_m2"].quantile(0.99)

    raw_price_plot = raw_df["price_eur"].dropna().clip(upper=price_upper) / 1000
    cleaned_price_plot = cleaned_df["price_eur"].dropna().clip(upper=price_upper) / 1000
    raw_area_plot = raw_df["area_m2"].dropna().clip(upper=area_upper)
    cleaned_area_plot = cleaned_df["area_m2"].dropna().clip(upper=area_upper)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    sns.histplot(raw_price_plot, bins=45, color="tab:blue", ax=axes[0, 0])
    axes[0, 0].set_title("Cena pre čišćenja")
    axes[0, 0].set_xlabel("Cena (hiljade EUR)")
    axes[0, 0].set_ylabel("Broj oglasa")

    sns.histplot(cleaned_price_plot, bins=45, color="tab:green", ax=axes[0, 1])
    axes[0, 1].set_title("Cena posle čišćenja")
    axes[0, 1].set_xlabel("Cena (hiljade EUR)")
    axes[0, 1].set_ylabel("Broj oglasa")

    sns.histplot(raw_area_plot, bins=45, color="tab:blue", ax=axes[1, 0])
    axes[1, 0].set_title("Kvadratura pre čišćenja")
    axes[1, 0].set_xlabel("Kvadratura (m²)")
    axes[1, 0].set_ylabel("Broj oglasa")

    sns.histplot(cleaned_area_plot, bins=45, color="tab:green", ax=axes[1, 1])
    axes[1, 1].set_title("Kvadratura posle čišćenja")
    axes[1, 1].set_xlabel("Kvadratura (m²)")
    axes[1, 1].set_ylabel("Broj oglasa")
    fig.tight_layout()
    render_matplotlib_figure(fig)


def render_city_charts(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> None:
    """Recreate the analysis notebook's city distribution charts and tables."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    raw_df["city"].value_counts().head(10).plot(
        kind="bar",
        ax=axes[0],
        color="tab:blue",
    )
    axes[0].set_title("Top 10 gradova - pre čišćenja")
    axes[0].set_xlabel("Grad")
    axes[0].set_ylabel("Broj oglasa")
    axes[0].tick_params(axis="x", rotation=45)

    cleaned_df["city"].value_counts().head(10).plot(
        kind="bar",
        ax=axes[1],
        color="tab:green",
    )
    axes[1].set_title("Top 10 gradova - posle čišćenja")
    axes[1].set_xlabel("Grad")
    axes[1].set_ylabel("Broj oglasa")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    render_matplotlib_figure(fig)

    city_before_after = (
        pd.DataFrame(
            {
                "Pre čišćenja": raw_df["city"].value_counts(),
                "Posle čišćenja": cleaned_df["city"].value_counts(),
            }
        )
        .fillna(0)
        .astype(int)
    )
    city_before_after["uklonjeno"] = (
        city_before_after["Pre čišćenja"] - city_before_after["Posle čišćenja"]
    )
    city_percentages = (
        (cleaned_df["city"].value_counts(normalize=True) * 100)
        .round(2)
        .rename("Procenat (%)")
        .to_frame()
    )
    city_percentages["Broj oglasa"] = cleaned_df["city"].value_counts()

    table_col1, table_col2 = st.columns(2)
    with table_col1:
        st.caption("Gradovi: pre/posle čišćenja")
        st.dataframe(
            city_before_after.sort_values("Pre čišćenja", ascending=False).head(10),
            width="stretch",
        )
    with table_col2:
        st.caption("Udeo gradova u očišćenom skupu")
        st.dataframe(city_percentages.head(15), width="stretch")


def render_price_per_m2_charts(
    cleaned_df: pd.DataFrame,
    model_df: pd.DataFrame,
) -> None:
    """Recreate the analysis notebook's price-per-m2 and regional summaries."""
    top_cleaned_cities = cleaned_df["city"].value_counts().head(8).index
    price_per_m2_by_city = model_df[model_df["city"].isin(top_cleaned_cities)].copy()
    price_per_m2_by_city["price_per_m2_plot"] = price_per_m2_by_city[
        "price_per_m2"
    ].clip(upper=price_per_m2_by_city["price_per_m2"].quantile(0.98))

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=price_per_m2_by_city, x="city", y="price_per_m2_plot", ax=ax)
    ax.set_title("Cena po m² za najčešće gradove posle čišćenja")
    ax.set_xlabel("Grad")
    ax.set_ylabel("Cena po m² (EUR, ograničeno na 98. percentil)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    render_matplotlib_figure(fig)

    region_price_summary = (
        cleaned_df[cleaned_df["city"].isin(["Beograd", "Novi Sad"])]
        .groupby(["city", "region"])["price_eur"]
        .agg(
            broj_oglasa="count",
            p05=lambda series: series.quantile(0.05),
            p25=lambda series: series.quantile(0.25),
            medijan="median",
            p75=lambda series: series.quantile(0.75),
            p95=lambda series: series.quantile(0.95),
        )
        .query("broj_oglasa >= 30")
        .sort_values(["city", "medijan"], ascending=[True, False])
        .round(0)
        .reset_index()
    )
    st.caption("Vodeći regioni po medijani cene za Beograd i Novi Sad")
    st.dataframe(
        region_price_summary.groupby("city").head(10),
        width="stretch",
    )


def render_data_tab() -> None:
    """Render EDA charts recreated from notebooks/analysis.ipynb."""
    st.subheader("Podaci")
    st.write(
        "Grafici su rekreirani iz `notebooks/analysis.ipynb` nad istim raw, očišćenim i feature-engineered skupovima."
    )
    with st.spinner("Učitavanje i priprema podataka..."):
        raw_df, cleaned_df, model_df, cleaning_summary = cached_modeling_data()

    col1, col2, col3 = st.columns(3)
    col1.metric("Raw oglasi", f"{len(raw_df):,}".replace(",", "."))
    col2.metric("Očišćeni oglasi", f"{len(cleaned_df):,}".replace(",", "."))
    col3.metric("Uklonjeno", f"{cleaning_summary['rows_removed_total']:,}".replace(",", "."))

    with st.expander("Sažetak čišćenja podataka", expanded=False):
        st.dataframe(
            pd.DataFrame([cleaning_summary]).T.rename(columns={0: "vrednost"}),
            width="stretch",
        )

    st.markdown("### Distribucije pre i posle čišćenja")
    render_distribution_charts(raw_df, cleaned_df)

    st.markdown("### Kategoričke raspodele i zastupljenost gradova")
    render_city_charts(raw_df, cleaned_df)

    st.markdown("### Cena po kvadratu i regionalni obrasci")
    render_price_per_m2_charts(cleaned_df, model_df)


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
    st.dataframe(metrics_df.round(3), width="stretch")

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


def render_header() -> None:
    """Render the dashboard hero/header."""
    st.markdown(
        """
        <style>
        .dashboard-hero {
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 1rem;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(16, 185, 129, 0.10));
        }
        .dashboard-hero h1 {
            margin: 0;
            font-size: 2.1rem;
            line-height: 1.2;
        }
        .dashboard-hero p {
            margin: 0.35rem 0 0 0;
            color: rgb(100, 116, 139);
            font-size: 1.02rem;
        }
        div[data-testid="stSegmentedControl"] {
            margin-bottom: 1rem;
        }
        </style>
        <div class="dashboard-hero">
            <h1>🏠 Procena cena nekretnina</h1>
            <p>CS490 dashboard za predikciju cena nekretnina u Srbiji.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_navigation() -> str:
    """Render tab-like navigation without eagerly rendering every section."""
    section_labels = {
        "Procena cene": "🏠 Procena cene",
        "Podaci": "📊 Podaci",
        "Modeli": "🤖 Modeli",
        "O projektu": "ℹ️ O projektu",
    }
    section = st.segmented_control(
        "Sekcija",
        options=list(section_labels),
        default="Procena cene",
        format_func=lambda option: section_labels[option],
        label_visibility="collapsed",
        width="stretch",
    )
    return section or "Procena cene"


def main() -> None:
    """Render the Streamlit dashboard."""
    render_header()

    try:
        pipeline, metadata = cached_prediction_artifact()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("uv run python -m src.training", language="bash")
        st.stop()

    section = render_section_navigation()

    if section == "Procena cene":
        render_prediction_tab(metadata, pipeline)
    elif section == "Podaci":
        render_data_tab()
    elif section == "Modeli":
        render_models_tab(metadata)
    else:
        render_project_tab(metadata)


if __name__ == "__main__":
    main()
