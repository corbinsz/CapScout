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
from models.artifacts import available, load_artifact
from models.occupancy import occupancy_available, predict_stay
from models.score import score_features
from streamlit_folium import st_folium

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Labels the amenity regexes actually match. The string is the listing's
# amenity list, not a second encoding.
AMENITY_CHOICES = {
    "Air conditioning": "Air conditioning",
    "Parking": "Free parking on premises",
    "Washer": "Washer",
    "Dryer": "Dryer",
    "Pool": "Pool",
    "Hot tub": "Hot tub",
    "Gym": "Gym",
    "Waterfront": "Waterfront",
    "Kitchen": "Kitchen",
    "Workspace": "Dedicated workspace",
    "Fireplace": "Fireplace",
    "Dishwasher": "Dishwasher",
    "Self check-in": "Self check-in",
    "Patio or balcony": "Patio or balcony",
    "BBQ": "BBQ grill",
    "EV charger": "EV charger",
    "Pets allowed": "Pets allowed",
    "Elevator": "Elevator",
}

INTERVAL_NOTE = {
    "log_conformal": (
        "The band uses one percentage width for every home, taken from how far "
        "off the model was on held-out blocks of the city."
    ),
    "scaled_log": (
        "The band is wider where the model is usually less sure. That is mostly "
        "the expensive listings."
    ),
    "asymmetric": (
        "The band can be wider above the estimate than below it, because the "
        "model tends to sit a bit low."
    ),
    "mondrian_asymmetric": (
        "The band is fit separately for cheaper and more expensive estimates, "
        "and it can be wider above the estimate than below it."
    ),
    "cqr": (
        "The band comes from a low estimate and a high estimate, adjusted so the "
        "held-out blocks were covered about 80 percent of the time."
    ),
}

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
      .block-container { padding-top: 1.2rem; max-width: 1180px; }
      .wordmark { font-family: Fraunces, Georgia, serif; font-size: 2.15rem; letter-spacing: -0.03em; margin: 0; }
      .eyebrow { letter-spacing: 0.14em; text-transform: uppercase; font-size: 0.72rem; color: #6d6456; margin: 0 0 0.2rem; }
      .card { background: #fffdf8; border: 1px solid #e4d8c4; border-radius: 18px; padding: 1.15rem 1.25rem 1.2rem; }
      .stay-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.8rem; margin: 0.8rem 0; }
      @media (max-width: 860px) { .stay-grid { grid-template-columns: 1fr; } }
      .price { font-family: Fraunces, Georgia, serif; font-size: 3.5rem; line-height: 0.95; letter-spacing: -0.04em; margin: 0.1rem 0 0.35rem; }
      .price span { font-family: Outfit, sans-serif; font-size: 1rem; color: #6d6456; margin-left: 0.35rem; letter-spacing: 0; }
      .muted { color: #6d6456; font-size: 0.92rem; line-height: 1.45; }
      .reason { display: grid; grid-template-columns: 1fr auto; gap: 0.15rem 0.8rem; margin: 0.7rem 0 0.1rem; }
      .bar { height: 7px; background: #efe6d6; border-radius: 99px; grid-column: 1 / -1; }
      .up { color: #0f6e56; font-weight: 600; }
      .down { color: #a33b24; font-weight: 600; }
      .comp { display: flex; justify-content: space-between; gap: 1rem; padding: 0.55rem 0; border-top: 1px solid #efe6d6; }
      .pill { display: inline-block; background: #e7f3ef; color: #0f6e56; border-radius: 99px; padding: 0.15rem 0.55rem; font-size: 0.78rem; font-weight: 600; }
      .track { position: relative; height: 10px; background: #efe6d6; border-radius: 99px; margin: 0.85rem 0 0.35rem; }
      .track i { position: absolute; top: 0; bottom: 0; background: #c9ddd4; border-radius: 99px; }
      .track b { position: absolute; top: -4px; width: 14px; height: 18px; margin-left: -7px; background: #0f6e56; border-radius: 4px; }
      .ends { display: flex; justify-content: space-between; font-size: 0.82rem; color: #6d6456; }
      h1, h2, h3 { font-family: Fraunces, Georgia, serif; letter-spacing: -0.03em; }
      div[data-testid="stMetric"] { background: #fffdf8; border: 1px solid #e4d8c4; border-radius: 16px; padding: 0.7rem 0.9rem; }
      div[data-testid="stExpander"] { background: #fffdf8; border-radius: 12px; }
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


@st.cache_data(show_spinner=False)
def _occupancy_metrics() -> dict | None:
    path = REPORTS / "occupancy_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_resource(show_spinner=False)
def _model(city: str):
    if not available(city):
        return None
    return load_artifact(city)


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
        '<p class="muted">An estimate of the advertised nightly rate, how many nights a year '
        "a similar listing was booked on a review-based estimate, the gross those two imply, "
        "and a grade of that gross inside the city. Not a quote, not a booking count, "
        "and not a profit.</p>",
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
            st.session_state.pop("scored", None)
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
            ["Not specified", "House", "Condo", "Townhouse", "Guest unit", "Cabin"],
        )
        knows_price = st.checkbox("Add a purchase price for a rough gross yield")
        home_price = None
        if knows_price:
            home_price = float(
                st.number_input("Purchase price ($)", min_value=10000, value=500000, step=5000)
            )
        lat = st.number_input("Latitude", value=float(st.session_state.lat), format="%.5f")
        lon = st.number_input("Longitude", value=float(st.session_state.lon), format="%.5f")
        st.session_state.lat = float(lat)
        st.session_state.lon = float(lon)
        extras = _optional_inputs(city)
        go = st.button("Estimate this home", type="primary", use_container_width=True)
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
        payload = {
            "latitude": float(st.session_state.lat),
            "longitude": float(st.session_state.lon),
            "bedrooms": int(bedrooms),
            "bathrooms": float(bathrooms),
            "accommodates": int(accommodates),
            "room_type": room,
        }
        if property_type != "Not specified":
            payload["property_type"] = property_type
        payload.update(extras["features"])
        scored = score_features(payload, city)
        if occupancy_available(city):
            scored["stay"] = predict_stay(
                payload,
                city,
                nightly_rate=float(scored["nightly_rate"]),
                purchase_price=home_price,
            )
        else:
            scored["stay"] = None
        st.session_state.scored = scored
        st.session_state.scored_city = city
        st.session_state.shown_comps = comps
        st.session_state.unknowns = extras["unknowns"]
    if st.session_state.get("scored") and st.session_state.get("scored_city") == city:
        _stay_block(city, st.session_state.scored.get("stay"))
        _result(
            st.session_state.scored,
            st.session_state.get("shown_comps", comps),
            st.session_state.get("unknowns") or [],
        )
    else:
        st.markdown(
            '<p class="muted" style="margin-top:0.6rem">Drop a pin or locate an address, '
            "then estimate. The rate is for a home that has not been listed, unless you "
            "add a description or reviews below. The nights estimate does not use reviews.</p>",
            unsafe_allow_html=True,
        )


def _optional_inputs(city: str) -> dict:
    """Fields the model can use, shown only when this city's model has them.

    Left blank, they stay missing. A zero would claim the home has no kitchen
    and no reviews, which is a different home from one we have not described.
    """
    features: dict = {}
    unknowns = ["listing text", "amenities", "reviews", "host"]
    model = _model(city)
    if model is None:
        return {"features": features, "unknowns": unknowns}
    columns = set(model.get("columns") or [])
    wants_text = any(name.startswith("text_") or name == "log_text_chars" for name in columns)
    wants_amenity = any(name.startswith("has_") or name == "amenity_count" for name in columns)
    wants_reviews = "review_score" in columns or "reviews_per_month" in columns
    wants_host = "is_superhost" in columns
    if not (wants_text or wants_amenity or wants_reviews or wants_host):
        return {"features": features, "unknowns": []}

    with st.expander("If this home already has a listing"):
        st.caption(
            "These are optional. Leave them blank for a home that has never been listed. "
            "The test scores are for ads where the fields were filled in."
        )
        if wants_text:
            text = st.text_area(
                "Title and description",
                placeholder="Sunny 2-bedroom with a downtown view, a short walk to the river.",
                height=90,
            )
            if text.strip():
                features["listing_text"] = text.strip()
                unknowns = [item for item in unknowns if item != "listing text"]
        if wants_amenity:
            known = st.checkbox("I know the amenities")
            if known:
                picked = st.multiselect("Amenities", list(AMENITY_CHOICES))
                quoted = ", ".join(f'"{AMENITY_CHOICES[name]}"' for name in picked)
                features["amenities"] = f"[{quoted}]"
                unknowns = [item for item in unknowns if item != "amenities"]
        if wants_reviews:
            reviews = st.selectbox(
                "Reviews",
                ["Unknown", "No reviews yet", "Has reviews"],
            )
            if reviews == "No reviews yet":
                features["number_of_reviews"] = 0
                features["reviews_per_month"] = 0.0
                unknowns = [item for item in unknowns if item != "reviews"]
            elif reviews == "Has reviews":
                r1, r2 = st.columns(2)
                features["review_score"] = float(
                    r1.number_input(
                        "Review score", min_value=1.0, max_value=5.0, value=4.8, step=0.1
                    )
                )
                features["reviews_per_month"] = float(
                    r2.number_input(
                        "Reviews per month", min_value=0.0, max_value=20.0, value=1.0, step=0.1
                    )
                )
                features["number_of_reviews"] = max(
                    1, int(round(features["reviews_per_month"] * 12))
                )
                unknowns = [item for item in unknowns if item != "reviews"]
        if wants_host:
            host = st.selectbox("Superhost", ["Unknown", "Yes", "No"])
            if host != "Unknown":
                features["host_is_superhost"] = "t" if host == "Yes" else "f"
                unknowns = [item for item in unknowns if item != "host"]
    return {"features": features, "unknowns": unknowns}


def _map(city: str, lat: float, lon: float, comps: pd.DataFrame) -> None:
    # OpenStreetMap's standard tiles do not need an API key.
    fmap = folium.Map(location=[lat, lon], zoom_start=13, tiles="OpenStreetMap", control_scale=True)
    folium.CircleMarker(
        [lat, lon],
        radius=9,
        color="#0f6e56",
        fill=True,
        fill_color="#0f6e56",
        fill_opacity=0.9,
        tooltip="This home",
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
    st.caption(
        "Click the map to move the pin. Coral dots are nearby Inside Airbnb ads, not model inputs."
    )


def _nights_note(city: str) -> str:
    payload = _occupancy_metrics()
    if not payload or city not in payload.get("cities", {}):
        return (
            "This is a model of a review-based estimate, not a count of bookings. "
            "The holdout report is not loaded, so the error is not shown here."
        )
    hold = payload["cities"][city]["holdout"]
    model = hold["model"]
    baseline = hold["baseline_area"]["mae"]
    if hold["cleared_bar"]:
        verdict = (
            "That cleared the pre-set bar: the average miss is at least 10% lower "
            "than the neighbourhood median, and R² is at least 0.10."
        )
    else:
        verdict = (
            "That did not clear the pre-set bar. The bar was a 10% lower average "
            "miss than the neighbourhood median, and an R² of at least 0.10. "
            "Read the number as a rough comparison, not a forecast."
        )
    klass = hold.get("classification") or {}
    class_bit = ""
    if klass.get("accuracy") is not None:
        if klass.get("cleared_bar"):
            class_bit = (
                " A separate three-way model, not this point estimate, was right "
                f"{klass['accuracy']:.0%} of the time on those blocks, against "
                f"{klass['baseline_accuracy']:.0%} for the neighbourhood's usual band, "
                "and that gap cleared its own bar."
            )
        else:
            class_bit = (
                " A separate three-way model, not this point estimate, was right "
                f"{klass['accuracy']:.0%} of the time, against {klass['baseline_accuracy']:.0%} "
                "for the neighbourhood's usual band. That did not clear its bar either."
            )
    return (
        "This is a model of Inside Airbnb's review-based estimate, not a count of bookings. "
        f"On held-out blocks of this city the average miss was {model['mae']:.0f} nights "
        f"(R² {model['r2']:.2f}). The neighbourhood median misses by {baseline:.0f} nights. "
        f"{verdict}{class_bit}"
    )


def _stay_block(city: str, stay: dict | None) -> None:
    if not stay:
        st.markdown(
            '<p class="muted">No nights model is saved for this city. '
            "From the repo root: python -m evaluation.run_occupancy</p>",
            unsafe_allow_html=True,
        )
        return
    nights = float(stay["booked_nights"])
    revenue = stay.get("annual_revenue")
    grade = stay.get("grade") or "—"
    words = stay.get("grade_words") or ""
    revenue_text = _money(revenue) if revenue is not None else "—"
    area = stay.get("area_median_nights")
    area_bit = ""
    if area is not None and stay.get("area_name"):
        area_bit = f" The neighbourhood median for {stay['area_name']} is about {area:.0f} nights."
    elif area is not None:
        area_bit = f" The citywide median of reviewed listings is about {area:.0f} nights."
    yield_bit = ""
    gross = stay.get("gross_yield")
    if gross is not None:
        yield_bit = (
            f" Rough gross yield {gross:.1%} on the purchase price you entered. "
            "That is predicted revenue divided by that price, before mortgage, taxes, "
            "fees, repairs, and any rule that limits short-term rentals."
        )
    band = stay.get("occupancy_class")
    band_bit = f" That point estimate is in the {band.lower()} band for this city." if band else ""
    st.markdown(
        '<div class="stay-grid">'
        '<div class="card"><p class="eyebrow">Booked nights</p>'
        f'<p class="price" style="font-size:2.4rem">{nights:.0f}<span>nights / year</span></p>'
        f'<p class="muted">Review-based estimate.{band_bit}{area_bit}</p></div>'
        '<div class="card"><p class="eyebrow">Estimated annual revenue</p>'
        f'<p class="price" style="font-size:2.4rem">{revenue_text}</p>'
        '<p class="muted">Predicted rate times predicted nights. Not what a host would clear.</p></div>'
        '<div class="card"><p class="eyebrow">Grade in this city</p>'
        f'<p class="price" style="font-size:2.4rem">{grade}<span>{words}</span></p>'
        '<p class="muted">Where that gross sits among reviewed listings here. '
        "Not a profit grade.</p></div>"
        "</div>"
        f'<p class="muted">{_nights_note(city)}{yield_bit} '
        "The grade does not know the mortgage, taxes, fees, or whether a short-term "
        "rental is legal at this address.</p>",
        unsafe_allow_html=True,
    )


def _result(scored: dict, comps: pd.DataFrame, unknowns: list[str]) -> None:
    low, high = scored["interval_low"], scored["interval_high"]
    rate = float(scored["nightly_rate"])
    note = ""
    if not scored.get("inside_extract", True):
        note = " This pin is outside the mapped extract, so nearby-place features are capped."
    if scored.get("census_join", 0) != 0:
        note += " This pin did not land in a tract polygon, so census figures are a nearby or city median."
    method = scored.get("interval_method") or "log_conformal"
    band = INTERVAL_NOTE.get(method, INTERVAL_NOTE["log_conformal"])
    if unknowns:
        missing = ", ".join(unknowns)
        band += f" Left blank, so treated as unknown: {missing}."
    band += note

    left, right = st.columns([1.08, 0.92], gap="large")
    with left:
        if low is None or high is None:
            track = ""
            interval = "interval unavailable"
        else:
            span = max(high - low, 1.0)
            mark = min(100.0, max(0.0, 100.0 * (rate - low) / span))
            track = (
                '<div class="track"><i style="left:0;right:0"></i>'
                f'<b style="left:{mark:.1f}%"></b></div>'
                f'<div class="ends"><span>{_money(low)}</span><span>{_money(high)}</span></div>'
            )
            interval = f"{_money(low)} – {_money(high)}"
        st.markdown(
            '<div class="card">'
            '<p class="eyebrow">Estimated advertised rate</p>'
            f'<p class="price">{_money(rate)}<span>per night</span></p>'
            f'<p class="muted">80% interval {interval}.</p>'
            f"{track}"
            f'<p class="muted" style="margin-top:0.7rem">{band} '
            "80% is the target on the held-out blocks, not a promise for this address.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        reasons = scored.get("reasons") or []
        scale = max((abs(item["shap"]) for item in reasons), default=1.0) or 1.0
        blocks = []
        for item in reasons:
            width = max(6, int(100 * abs(item["shap"]) / scale))
            klass = "up" if item["shap"] >= 0 else "down"
            color = "#0f6e56" if item["shap"] >= 0 else "#a33b24"
            sign = "+" if item["approx_percent"] >= 0 else "−"
            sentence = item.get("sentence") or pretty(item["feature"])
            blocks.append(
                f'<div class="reason"><span>{sentence}</span>'
                f'<span class="{klass}">{sign}{abs(item["approx_percent"]):.0f}%</span>'
                f'<div class="bar"><span style="display:block;height:7px;border-radius:99px;'
                f'width:{width}%;background:{color}"></span></div></div>'
            )
        st.markdown(
            '<div class="card" style="margin-top:0.9rem"><p class="eyebrow">Why this rate</p>'
            + (
                "".join(blocks)
                if blocks
                else '<p class="muted">No single feature moved this estimate much.</p>'
            )
            + '<p class="muted" style="margin-top:0.8rem">The percent is exp(SHAP) − 1 on the log-price model, '
            "an approximate share of this prediction. It was not used to pick the model.</p></div>",
            unsafe_allow_html=True,
        )
    with right:
        prices = [float(row.price_clean) for row in comps.itertuples(index=False)]
        median = float(pd.Series(prices).median()) if prices else rate
        gap = rate - median
        if abs(gap) < 5:
            compare = "The estimate sits about on the median of these five."
        elif gap > 0:
            compare = f"The estimate is {_money(gap)} above the median of these five."
        else:
            compare = f"The estimate is {_money(abs(gap))} below the median of these five."
        rows = []
        for row in comps.itertuples(index=False):
            neighbourhood = str(row.neighbourhood_cleansed).replace("<", "")
            baths = row.bathrooms
            bath_bit = f" · {baths:g} ba" if pd.notna(baths) else ""
            rows.append(
                f'<div class="comp"><div><strong>{_money(row.price_clean)}</strong>'
                f'<div class="muted">{neighbourhood}</div></div>'
                f'<div style="text-align:right"><span class="pill">{row.km:.1f} km</span>'
                f'<div class="muted">{int(row.bedrooms)} bd{bath_bit}</div></div></div>'
            )
        st.markdown(
            '<div class="card"><p class="eyebrow">Nearby advertised rates</p>'
            '<p class="muted">Same room type when there are enough, within one bedroom. '
            "These ads are not fed into the model. They are the comps a person would look up.</p>"
            f'<p class="muted" style="margin-top:0.4rem">{compare} Median {_money(median)}.</p>'
            + "".join(rows)
            + "</div>",
            unsafe_allow_html=True,
        )


def _headline_block(city_report: dict) -> tuple[dict, str]:
    """The final holdout is the result. Seed 7 remains as a comparison."""
    if "final_holdout" in city_report:
        return city_report["final_holdout"], "Final holdout · scored once"
    return city_report["locked_test"], "Locked spatial test · scored once"


def _results_page() -> None:
    payload = _metrics()
    if not payload or payload.get("smoke"):
        st.markdown('<p class="eyebrow">Holdout · scored once</p>', unsafe_allow_html=True)
        st.markdown('<p class="wordmark">What the model actually did</p>', unsafe_allow_html=True)
        st.info("No full report yet. From the repo root: python -m evaluation.run_eval")
        return
    cities = payload["study_cities"]
    _block, eyebrow = _headline_block(payload["cities"][cities[0]])
    st.markdown(f'<p class="eyebrow">{eyebrow}</p>', unsafe_allow_html=True)
    st.markdown('<p class="wordmark">What the model actually did</p>', unsafe_allow_html=True)
    cards = []
    for city in cities:
        report_block, _label = _headline_block(payload["cities"][city])
        report = report_block["gbm"]
        coverage = report_block["interval"]["coverage"]
        cards.append(
            '<div class="card">'
            f"<p class=\"eyebrow\">{payload['cities'][city]['display']}</p>"
            f"<p class=\"price\" style=\"font-size:2.1rem\">{report['r2']:.3f}</p>"
            f"<p class=\"muted\">R² · MAE {_money(report['mae'])} · the 80% band covered {coverage:.1%}</p>"
            "</div>"
        )
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;margin:0.4rem 0 1rem">'
        + "".join(cards)
        + "</div>",
        unsafe_allow_html=True,
    )
    rows = []
    for city in cities:
        locked, _label = _headline_block(payload["cities"][city])
        labels = [
            ("gbm", "This round"),
            ("v3_refit", "v3 specification, refit"),
            ("v2_refit", "v2 specification, refit"),
            ("previous_spec", "Size and location only"),
            ("ridge", "Ridge, same columns"),
            ("baseline_room", "Neighbourhood × room"),
        ]
        for key, label in labels:
            if key not in locked:
                continue
            metrics = locked[key]
            rows.append(
                {
                    "City": payload["cities"][city]["display"],
                    "Model": label,
                    "R²": f"{metrics['r2']:.3f}",
                    "MAE": f"{metrics['mae']:.2f}",
                    "RMSE": f"{metrics['rmse']:.2f}",
                    "MAPE %": f"{metrics['mape']:.1f}",
                    "Median APE %": f"{metrics['median_ape']:.1f}",
                }
            )
    st.table(pd.DataFrame(rows))
    protocol = payload["protocol"]
    coverage = []
    for city in cities:
        locked, _label = _headline_block(payload["cities"][city])
        interval = locked["interval"]
        row = {
            "City": payload["cities"][city]["display"],
            "Method": interval.get("method", "log_conformal").replace("_", " "),
            "Coverage": f"{interval['coverage']:.1%}",
            "Median width": _money(interval["median_width"]),
        }
        methods = locked.get("interval_methods") or {}
        for name, label in (
            ("scaled_log", "Scaled band"),
            ("asymmetric", "Asymmetric"),
            ("mondrian_asymmetric", "By price band"),
            ("log_conformal", "Log band"),
            ("cqr", "Conformalized quantiles"),
        ):
            if name in methods:
                row[label] = f"{methods[name]['coverage']:.1%}"
        coverage.append(row)
    st.markdown("##### Prediction intervals")
    st.table(pd.DataFrame(coverage))
    st.caption(
        f"Selected {protocol.get('feature_set', protocol.get('chosen_feature_set'))} · "
        f"{protocol.get('library')} · depth {protocol.get('chosen_depth')}. "
        "The band in the estimator is the one chosen on the development folds. "
        "The other coverages were scored once and were not used to switch methods."
    )
    if any("final_holdout" in payload["cities"][city] for city in cities):
        st.markdown("##### Seed 7, the previous locked test")
        st.caption(
            "Seed 7 was already published, so it was not used to choose anything. "
            "These rows are the same listings as v3."
        )
        seed_rows = []
        for city in cities:
            block = payload["cities"][city].get("locked_test")
            if not block or "v3_refit" not in block:
                continue
            seed_rows.append(
                {
                    "City": payload["cities"][city]["display"],
                    "This round R²": f"{block['gbm']['r2']:.3f}",
                    "This round MAE": _money(block["gbm"]["mae"]),
                    "v3 MAE": _money(block["v3_refit"]["mae"]),
                    "Baseline MAE": _money(block["baseline_room"]["mae"]),
                    "80% coverage": f"{block['interval']['coverage']:.1%}",
                }
            )
        if seed_rows:
            st.table(pd.DataFrame(seed_rows))
    if any("confirmation" in payload["cities"][city] for city in cities):
        st.markdown("##### Seed 11")
        st.caption(payload.get("holdout_note") or payload.get("confirmation_note") or "")
        confirm_rows = []
        for city in cities:
            block = payload["cities"][city].get("confirmation")
            if not block:
                continue
            previous = block.get("v3_refit") or block.get("v2_refit")
            confirm_rows.append(
                {
                    "City": payload["cities"][city]["display"],
                    "Listings": f"{block['gbm']['n']:,}",
                    "This round R²": f"{block['gbm']['r2']:.3f}",
                    "This round MAE": _money(block["gbm"]["mae"]),
                    "Previous spec MAE": _money(previous["mae"]),
                    "Interval coverage": f"{block['interval']['coverage']:.1%}",
                }
            )
        st.table(pd.DataFrame(confirm_rows))
    occ = _occupancy_metrics()
    if occ and not occ.get("smoke"):
        st.markdown("##### Booked nights, seed 31")
        st.caption(
            "A separate holdout from the price model. The target is a review-based "
            "estimate of nights booked in the last year, not a count of bookings. "
            "The neighbourhood median is the baseline."
        )
        occ_rows = []
        for city in occ["cities"]:
            hold = occ["cities"][city]["holdout"]
            occ_rows.append(
                {
                    "City": occ["cities"][city]["display"],
                    "R²": f"{hold['model']['r2']:.3f}",
                    "MAE nights": f"{hold['model']['mae']:.1f}",
                    "Area median MAE": f"{hold['baseline_area']['mae']:.1f}",
                    "Cleared bar": "Yes" if hold["cleared_bar"] else "No",
                    "Revenue R²": f"{hold['revenue']['r2']:.3f}",
                    "Revenue MAE": _money(hold["revenue"]["mae"]),
                    "Class accuracy": f"{hold['classification']['accuracy']:.0%}",
                }
            )
        st.table(pd.DataFrame(occ_rows))
        st.caption(occ["bar"]["rule"])
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
        "Study", ["Estimate a home", "Holdout results"], label_visibility="collapsed"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "A personal project, built in spare time, on advertised Airbnb rates "
        "and a review-based nights estimate in Austin, San Diego, and Seattle. "
        "It runs on this machine only."
    )
    if page == "Estimate a home":
        _predict_page()
    else:
        _results_page()


main()
