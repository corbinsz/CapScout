"""Local study app. Binds to 127.0.0.1 via .streamlit/config.toml.

    streamlit run app/streamlit_app.py

Pick a city, drop a pin or search an address, and read a nightly-rate
estimate with an interval, the main reasons, and nearby advertised rates.
The model-results page is the locked test, not a second scoreboard.

This file does not deploy anything and does not call a listing site.
Addresses go to the Census geocoder, which is a public-domain service.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import folium
import pandas as pd
import requests
import streamlit as st
from data.cities import CITIES, STUDY_CITIES
from data.ingest import load_listings
from features.columns import pretty
from features.geo import haversine_km
from models.artifacts import available
from models.score import score_features
from streamlit_folium import st_folium

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

st.set_page_config(
    page_title="CapScout",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,520;9..144,640&family=Outfit:wght@380;500;600&display=swap');
      html, body, [class*="css"] { font-family: Outfit, "Segoe UI", sans-serif; }
      .stApp { background: #f4efe6; color: #1c1915; }
      header[data-testid="stHeader"] { background: transparent; }
      #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
      [data-testid="stSidebar"] { background: #1c1915; }
      [data-testid="stSidebar"] * { color: #f4efe6 !important; }
      [data-testid="stSidebar"] .stRadio label { font-size: 1.02rem; }
      .block-container { padding-top: 1.4rem; max-width: 1180px; }
      .wordmark { font-family: Fraunces, Georgia, serif; font-size: 2rem; letter-spacing: -0.03em; margin: 0; }
      .eyebrow { letter-spacing: 0.14em; text-transform: uppercase; font-size: 0.72rem; color: #6d6456; margin: 0 0 0.2rem; }
      .card { background: #fffdf8; border: 1px solid #e4d8c4; border-radius: 18px; padding: 1.15rem 1.25rem 1.2rem; }
      .price { font-family: Fraunces, Georgia, serif; font-size: 3.4rem; line-height: 1; letter-spacing: -0.04em; margin: 0.15rem 0; }
      .price span { font-family: Outfit, sans-serif; font-size: 1rem; color: #6d6456; margin-left: 0.35rem; }
      .muted { color: #6d6456; font-size: 0.92rem; }
      .reason { display: grid; grid-template-columns: 1fr auto; gap: 0.15rem 0.8rem; margin: 0.55rem 0 0.15rem; }
      .bar { height: 7px; background: #efe6d6; border-radius: 99px; grid-column: 1 / -1; }
      .bar i { display: block; height: 7px; border-radius: 99px; }
      .up { color: #0f6e56; font-weight: 600; }
      .down { color: #a33b24; font-weight: 600; }
      .comp { display: flex; justify-content: space-between; gap: 1rem; padding: 0.55rem 0; border-top: 1px solid #efe6d6; }
      .pill { display: inline-block; background: #e7f3ef; color: #0f6e56; border-radius: 99px; padding: 0.15rem 0.55rem; font-size: 0.78rem; font-weight: 600; }
      h1, h2, h3 { font-family: Fraunces, Georgia, serif; letter-spacing: -0.03em; }
      div[data-testid="stMetric"] { background: #fffdf8; border: 1px solid #e4d8c4; border-radius: 16px; padding: 0.7rem 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _listings(city: str) -> pd.DataFrame:
    frame, _stats = load_listings(city)
    columns = [
        "latitude",
        "longitude",
        "price_clean",
        "bedrooms",
        "bathrooms",
        "accommodates",
        "room_type",
        "neighbourhood_cleansed",
    ]
    return frame[columns].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _metrics() -> dict | None:
    path = REPORTS / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _geocode(address: str, city_key: str) -> tuple[float, float, str] | None:
    city = CITIES[city_key]
    query = f"{address}, {city['name']}, {city['state']}"
    response = requests.get(
        "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress",
        params={"address": query, "benchmark": "Public_AR_Current", "format": "json"},
        timeout=20,
    )
    response.raise_for_status()
    matches = response.json().get("result", {}).get("addressMatches") or []
    if not matches:
        return None
    coords = matches[0]["coordinates"]
    return float(coords["y"]), float(coords["x"]), str(matches[0]["matchedAddress"])


def _comps(city: str, lat: float, lon: float, room: str, bedrooms: float) -> pd.DataFrame:
    frame = _listings(city).copy()
    frame["km"] = haversine_km(
        frame["latitude"].to_numpy(), frame["longitude"].to_numpy(), lat, lon
    )
    same = frame[(frame["room_type"] == room) & (frame["bedrooms"].sub(bedrooms).abs() <= 1)]
    pool = same if len(same) >= 5 else frame[frame["room_type"] == room]
    if len(pool) < 5:
        pool = frame
    return pool.nsmallest(5, "km")


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _predict_page() -> None:
    st.markdown('<p class="eyebrow">Personal study · local only</p>', unsafe_allow_html=True)
    st.markdown('<p class="wordmark">CapScout</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="muted">A nightly advertised rate for a listing you describe. '
        "Not a quote, not a booked price, and not a yield.</p>",
        unsafe_allow_html=True,
    )

    if "city" not in st.session_state:
        st.session_state.city = STUDY_CITIES[0]
        st.session_state.lat = float(CITIES[STUDY_CITIES[0]]["latitude"])
        st.session_state.lon = float(CITIES[STUDY_CITIES[0]]["longitude"])

    left, right = st.columns([0.92, 1.08], gap="large")
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        city = st.selectbox(
            "City",
            STUDY_CITIES,
            format_func=lambda key: CITIES[key]["display"],
            index=list(STUDY_CITIES).index(st.session_state.city),
        )
        if city != st.session_state.city:
            st.session_state.city = city
            st.session_state.lat = float(CITIES[city]["latitude"])
            st.session_state.lon = float(CITIES[city]["longitude"])
            st.session_state.pop("click_token", None)
            st.rerun()
        address = st.text_input("Address", placeholder="1100 Congress Ave")
        if st.button("Locate address", use_container_width=True):
            try:
                found = _geocode(address, city) if address.strip() else None
            except requests.RequestException:
                found = None
                st.error(
                    "The Census geocoder did not answer. Enter a latitude and longitude instead."
                )
            if found:
                st.session_state.lat, st.session_state.lon, label = found
                st.session_state.click_token = (
                    round(st.session_state.lat, 5),
                    round(st.session_state.lon, 5),
                )
                st.caption(label)
            elif address.strip():
                st.warning("No match in that city. Try a fuller street address, or drop a pin.")
        c1, c2, c3 = st.columns(3)
        bedrooms = c1.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1)
        bathrooms = c2.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
        accommodates = c3.number_input("Guests", min_value=1, max_value=16, value=4, step=1)
        room = st.selectbox(
            "Room type",
            ["Entire home/apt", "Private room", "Shared room", "Hotel room"],
        )
        property_type = st.selectbox(
            "Building",
            ["House", "Apartment", "Condo", "Townhouse", "Guest suite", "Other"],
        )
        lat = st.number_input("Latitude", value=float(st.session_state.lat), format="%.5f")
        lon = st.number_input("Longitude", value=float(st.session_state.lon), format="%.5f")
        st.session_state.lat = float(lat)
        st.session_state.lon = float(lon)
        go = st.button("Estimate nightly rate", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    comps = _comps(
        city, float(st.session_state.lat), float(st.session_state.lon), room, float(bedrooms)
    )
    with right:
        _map(city, float(st.session_state.lat), float(st.session_state.lon), comps)

    # The button is true only on the run that clicked it. The map widget
    # reruns the script on its own, so the estimate has to live in session
    # state or it disappears before anyone can read it.
    if go:
        if not available(city):
            st.error(
                "No saved model for this city. From the repo root: python -m evaluation.run_eval"
            )
            return
        st.session_state.scored = score_features(
            {
                "latitude": float(st.session_state.lat),
                "longitude": float(st.session_state.lon),
                "bedrooms": int(bedrooms),
                "bathrooms": float(bathrooms),
                "accommodates": int(accommodates),
                "room_type": room,
                "property_type": property_type,
            },
            city,
        )
        st.session_state.shown_comps = comps
    if st.session_state.get("scored"):
        _result(st.session_state.scored, st.session_state.get("shown_comps", comps))


def _map(city: str, lat: float, lon: float, comps: pd.DataFrame) -> None:
    # OpenStreetMap's standard tiles do not need an API key. Carto's
    # positron style started requiring one, which drew "key required"
    # across the map instead of streets.
    fmap = folium.Map(location=[lat, lon], zoom_start=13, tiles="OpenStreetMap", control_scale=True)
    folium.CircleMarker(
        [lat, lon],
        radius=9,
        color="#0f6e56",
        fill=True,
        fill_color="#0f6e56",
        fill_opacity=0.9,
        tooltip="This listing",
    ).add_to(fmap)
    for row in comps.itertuples(index=False):
        folium.CircleMarker(
            [row.latitude, row.longitude],
            radius=5,
            color="#a33b24",
            fill=True,
            fill_color="#a33b24",
            fill_opacity=0.75,
            tooltip=f"{_money(row.price_clean)} · {row.km:.1f} km",
        ).add_to(fmap)
    out = st_folium(
        fmap, height=460, width=None, returned_objects=["last_clicked"], key=f"map-{city}"
    )
    click = (out or {}).get("last_clicked")
    if click:
        token = (round(float(click["lat"]), 5), round(float(click["lng"]), 5))
        if token != st.session_state.get("click_token"):
            st.session_state.click_token = token
            st.session_state.lat = token[0]
            st.session_state.lon = token[1]
            st.rerun()
    st.caption("Click the map to drop a pin. Coral dots are nearby Inside Airbnb listings.")


def _result(scored: dict, comps: pd.DataFrame) -> None:
    low, high = scored["interval_low"], scored["interval_high"]
    note = ""
    if not scored.get("inside_extract", True):
        note = " This pin is outside the mapped extract, so nearby-place features are capped."
    if scored.get("census_join", 0) != 0:
        note += " This pin did not land in a tract polygon, so census figures are a nearby or city median."
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        interval = (
            f"{_money(low)} – {_money(high)}"
            if low is not None and high is not None
            else "interval unavailable"
        )
        st.markdown(
            '<div class="card">'
            '<p class="eyebrow">Predicted advertised rate</p>'
            f"<p class=\"price\">{_money(scored['nightly_rate'])}<span>per night</span></p>"
            f'<p class="muted">80% interval {interval}. Fitted on development listings only.{note}</p>'
            "</div>",
            unsafe_allow_html=True,
        )
        reasons = scored.get("reasons") or []
        scale = max((abs(item["shap"]) for item in reasons), default=1.0) or 1.0
        blocks = []
        for item in reasons:
            width = int(100 * abs(item["shap"]) / scale)
            klass = "up" if item["shap"] >= 0 else "down"
            color = "#0f6e56" if item["shap"] >= 0 else "#a33b24"
            sign = "+" if item["approx_percent"] >= 0 else "−"
            # No blank lines. Streamlit's markdown parser leaves HTML mode
            # at a blank line and then prints the remaining tags as text.
            label = pretty(item["feature"])
            value = item.get("value")
            # A zero flag is a fact ("this is not a private room"), not a missing field.
            if item["feature"].startswith(("is_", "has_")) and value is not None:
                label = f"{label}: {'yes' if value >= 0.5 else 'no'}"
            blocks.append(
                f'<div class="reason"><span>{label}</span>'
                f'<span class="{klass}">{sign}{abs(item["approx_percent"]):.0f}%</span>'
                f'<div class="bar"><span style="display:block;height:7px;border-radius:99px;'
                f'width:{width}%;background:{color}"></span></div></div>'
            )
        st.markdown(
            '<div class="card" style="margin-top:0.9rem"><p class="eyebrow">Why this rate</p>'
            + "".join(blocks)
            + '<p class="muted" style="margin-top:0.8rem">Percents are approximate: exp(SHAP) − 1 on the log-price model. '
            "They explain this prediction. They were not used to pick the model.</p></div>",
            unsafe_allow_html=True,
        )
    with right:
        rows = []
        for row in comps.itertuples(index=False):
            neighbourhood = str(row.neighbourhood_cleansed).replace("<", "")
            rows.append(
                f'<div class="comp"><div><strong>{_money(row.price_clean)}</strong>'
                f'<div class="muted">{neighbourhood}</div></div>'
                f'<div style="text-align:right"><span class="pill">{row.km:.1f} km</span>'
                f'<div class="muted">{int(row.bedrooms)} bd · {row.room_type}</div></div></div>'
            )
        st.markdown(
            '<div class="card"><p class="eyebrow">Nearby advertised rates</p>'
            '<p class="muted">Same room type when there are enough, within one bedroom. '
            "These are not fed into the model. They are the comps a person would look at.</p>"
            + "".join(rows)
            + "</div>",
            unsafe_allow_html=True,
        )


def _results_page() -> None:
    st.markdown('<p class="eyebrow">Locked spatial test · scored once</p>', unsafe_allow_html=True)
    st.markdown('<p class="wordmark">What the model actually did</p>', unsafe_allow_html=True)
    payload = _metrics()
    if not payload or payload.get("smoke"):
        st.info("No full report yet. From the repo root: python -m evaluation.run_eval")
        return
    cities = payload["study_cities"]
    cols = st.columns(len(cities))
    for column, city in zip(cols, cities, strict=True):
        report = payload["cities"][city]["locked_test"]["gbm"]
        column.metric(
            payload["cities"][city]["display"],
            f"R² {report['r2']:.3f}",
            delta=f"MAE {_money(report['mae'])} · MAPE {report['mape']:.1f}%",
            delta_color="off",
        )
    st.markdown("")
    rows = []
    for city in cities:
        locked = payload["cities"][city]["locked_test"]
        for key, label in (
            ("gbm", "Selected model"),
            ("previous_spec", "Previous specification"),
            ("ridge", "Ridge"),
            ("baseline_room", "Neighbourhood × room"),
        ):
            metrics = locked[key]
            rows.append(
                {
                    "City": payload["cities"][city]["display"],
                    "Model": label,
                    "R²": round(metrics["r2"], 3),
                    "MAE": round(metrics["mae"], 2),
                    "RMSE": round(metrics["rmse"], 2),
                    "MAPE %": round(metrics["mape"], 1),
                    "Median APE %": round(metrics["median_ape"], 1),
                }
            )
    st.table(pd.DataFrame(rows))
    protocol = payload["protocol"]
    coverage = [
        {
            "City": payload["cities"][city]["display"],
            "80% interval coverage": round(
                payload["cities"][city]["locked_test"]["interval"]["coverage"], 3
            ),
            "Median width": round(
                payload["cities"][city]["locked_test"]["interval"]["median_width"], 0
            ),
            "Quantile-model coverage": round(
                payload["cities"][city]["locked_test"]["quantile_interval"]["coverage"], 3
            ),
        }
        for city in cities
    ]
    st.markdown("##### Prediction intervals")
    st.table(pd.DataFrame(coverage))
    st.caption(
        f"Selected {protocol['chosen_feature_set']} · {protocol['library']} · depth {protocol['chosen_depth']}. "
        "The interval in the estimator is the conformal one. Quantile coverage is the other method, scored once."
    )
    figures = [
        ("pred_vs_actual.png", "Predicted versus actual"),
        ("residuals.png", "Residuals"),
        ("calibration.png", "Interval coverage by price"),
        ("shap.png", "What the model uses"),
        ("partial_dependence.png", "Partial dependence"),
        ("ablation_mae.png", "Ablation"),
        ("model_comparison.png", "Development comparison"),
        ("learning_curves.png", "Learning curves"),
    ]
    for name, title in figures:
        path = FIGURES / name
        if path.exists():
            st.markdown(f"##### {title}")
            st.image(str(path), use_container_width=True)


def main() -> None:
    page = st.sidebar.radio(
        "Study", ["Estimate a rate", "Locked-test results"], label_visibility="collapsed"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Built in spare time as a personal study of advertised Airbnb rates "
        "in Austin, San Diego, and Seattle. It runs on this machine only."
    )
    if page == "Estimate a rate":
        _predict_page()
    else:
        _results_page()


main()
