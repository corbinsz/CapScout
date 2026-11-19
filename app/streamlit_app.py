"""Local study app. Binds to 127.0.0.1 via .streamlit/config.toml.

    streamlit run app/streamlit_app.py

Pick a city, enter an address or a pin, and read a nightly rate with a range,
a nights estimate, yearly revenue, a grade, the main reasons, and nearby ads.
This file does not deploy anything.
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
from app.locate import LocateError, check_pin, locate_address
from data.cities import CITIES, STUDY_CITIES
from models.serve import load_comp_frame, nearby_comps, score_home
from streamlit_folium import st_folium

REPORTS = ROOT / "reports"
AMENITY_CHOICES = {
    "Air conditioning": "Air conditioning",
    "Parking": "Free parking on premises",
    "Washer": "Washer",
    "Kitchen": "Kitchen",
    "Pool": "Pool",
    "Hot tub": "Hot tub",
    "Waterfront": "Waterfront",
}

st.set_page_config(page_title="CapScout", page_icon="◈", layout="wide")
st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-family: "Segoe UI", sans-serif; }
      .stApp { background: #f4efe6; color: #1c1915; }
      header[data-testid="stHeader"] { background: transparent; }
      #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
      [data-testid="stSidebar"] { background: #1c1915; }
      [data-testid="stSidebar"] * { color: #f4efe6 !important; }
      .block-container { padding-top: 1.2rem; max-width: 1180px; }
      .wordmark { font-family: Georgia, serif; font-size: 2.15rem; margin: 0; }
      .eyebrow { letter-spacing: 0.14em; text-transform: uppercase; font-size: 0.72rem; color: #6d6456; margin: 0 0 0.2rem; }
      .card { background: #fffdf8; border: 1px solid #e4d8c4; border-radius: 18px; padding: 1.15rem 1.25rem 1.2rem; }
      .stay-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.8rem; margin: 0.8rem 0; }
      @media (max-width: 860px) { .stay-grid { grid-template-columns: 1fr; } }
      .price { font-family: Georgia, serif; font-size: 3.2rem; line-height: 0.95; margin: 0.1rem 0 0.35rem; }
      .price span { font-family: "Segoe UI", sans-serif; font-size: 1rem; color: #6d6456; margin-left: 0.35rem; }
      .muted { color: #6d6456; font-size: 0.92rem; line-height: 1.45; }
      .reason { margin: 0.65rem 0 0.1rem; }
      .comp { display: flex; justify-content: space-between; gap: 1rem; padding: 0.55rem 0; border-top: 1px solid #efe6d6; }
      .pill { display: inline-block; background: #e7f3ef; color: #0f6e56; border-radius: 99px; padding: 0.15rem 0.55rem; font-size: 0.78rem; font-weight: 600; }
      h1, h2, h3 { font-family: Georgia, serif; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _fetch(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def _comps_source(city: str) -> pd.DataFrame:
    return load_comp_frame(city)


@st.cache_data(show_spinner=False)
def _study() -> dict | None:
    path = REPORTS / "study_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _place_name(name: str) -> str:
    """A neighbourhood label. Austin stores ZIP codes in this field."""
    text = str(name).strip()
    if text.isdigit():
        return f"ZIP {text}"
    return text


def _predict_page() -> None:
    st.markdown(
        '<p class="eyebrow">Personal study · this computer only</p>', unsafe_allow_html=True
    )
    st.markdown('<p class="wordmark">CapScout</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="muted">A nightly rate, a range around it, a rough guess of nights booked '
        "in a year, the yearly revenue those two imply, and a letter grade for that revenue "
        "inside the city. Not a quote and not a count of bookings.</p>",
        unsafe_allow_html=True,
    )
    if "city" not in st.session_state:
        st.session_state.city = STUDY_CITIES[0]
        st.session_state.lat = float(CITIES[STUDY_CITIES[0]]["latitude"])
        st.session_state.lon = float(CITIES[STUDY_CITIES[0]]["longitude"])

    left, right = st.columns([0.92, 1.08], gap="large")
    with left:
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
            st.session_state.pop("scored", None)
            st.rerun()
        address = st.text_input("Address", placeholder="1100 Congress Ave")
        if st.button("Locate address", use_container_width=True):
            try:
                found = locate_address(address, city, _fetch)
            except LocateError as exc:
                st.warning(exc.message)
            else:
                st.session_state.lat = found["latitude"]
                st.session_state.lon = found["longitude"]
                st.caption(found["label"])
        c1, c2, c3 = st.columns(3)
        bedrooms = c1.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1)
        bathrooms = c2.number_input("Bathrooms", min_value=0.0, max_value=12.0, value=1.0, step=0.5)
        guests = c3.number_input("Guests", min_value=1, max_value=16, value=4, step=1)
        room = st.selectbox(
            "Room type",
            ["Entire home/apt", "Private room", "Shared room", "Hotel room"],
        )
        building = st.selectbox(
            "Building",
            ["Not specified", "House", "Condo", "Townhouse", "Guest unit", "Cabin"],
        )
        extras = _optional_inputs()
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
        go = st.button("Estimate this home", type="primary", use_container_width=True)

    comps = nearby_comps(
        _comps_source(city),
        float(st.session_state.lat),
        float(st.session_state.lon),
        room,
        float(bedrooms),
    )
    with right:
        _map(city, float(st.session_state.lat), float(st.session_state.lon), comps)

    if go:
        try:
            check_pin(float(lat), float(lon), city)
            spec = {
                "latitude": float(lat),
                "longitude": float(lon),
                "bedrooms": int(bedrooms),
                "bathrooms": float(bathrooms),
                "accommodates": int(guests),
                "room_type": room,
                "purchase_price": home_price,
            }
            if building != "Not specified":
                spec["property_type"] = building
            spec.update(extras)
            if len(comps):
                spec["neighbourhood"] = str(comps.iloc[0]["neighbourhood_cleansed"])
            st.session_state.scored = score_home(city, spec)
            st.session_state.scored_city = city
            st.session_state.shown_comps = comps
        except LocateError as exc:
            st.session_state.pop("scored", None)
            st.error(exc.message)
        except FileNotFoundError:
            st.error(
                "No saved model for this city. From the repo root: python -m evaluation.run_study"
            )

    if st.session_state.get("scored") and st.session_state.get("scored_city") == city:
        _stay(st.session_state.scored)
        _result(st.session_state.scored, st.session_state.get("shown_comps", comps))
    else:
        st.markdown(
            '<p class="muted">Locate an address or enter a latitude and longitude, then estimate. '
            "Leave amenities and reviews blank for a home that has never been listed. "
            "Blank fields use a typical value, not zero.</p>",
            unsafe_allow_html=True,
        )


def _optional_inputs() -> dict:
    features: dict = {}
    with st.expander("If this home already has a listing"):
        st.caption(
            "Optional. Leave these alone for a home that has never been listed. "
            "A blank is a typical value, not a zero."
        )
        if st.checkbox("I know the amenities"):
            picked = st.multiselect("Amenities", list(AMENITY_CHOICES))
            # The seven yes/no fields only. The total amenity count stays unknown.
            features["amenity_flags"] = {
                "has_ac": "Air conditioning" in picked,
                "has_parking": "Parking" in picked,
                "has_washer": "Washer" in picked,
                "has_kitchen": "Kitchen" in picked,
                "has_pool": "Pool" in picked,
                "has_hottub": "Hot tub" in picked,
                "has_waterfront": "Waterfront" in picked,
            }
        reviews = st.selectbox("Reviews", ["Unknown", "No reviews yet", "Has reviews"])
        if reviews == "No reviews yet":
            features["number_of_reviews"] = 0
        elif reviews == "Has reviews":
            r1, r2 = st.columns(2)
            features["review_score"] = float(
                r1.number_input("Review score", min_value=1.0, max_value=5.0, value=4.8, step=0.1)
            )
            features["number_of_reviews"] = int(
                r2.number_input("Number of reviews", min_value=1, max_value=1000, value=20, step=1)
            )
        host = st.selectbox("Superhost", ["Unknown", "Yes", "No"])
        if host != "Unknown":
            features["host_is_superhost"] = "t" if host == "Yes" else "f"
    return features


def _map(city: str, lat: float, lon: float, comps: pd.DataFrame) -> None:
    fmap = folium.Map(location=[lat, lon], zoom_start=13, tiles="OpenStreetMap", control_scale=True)
    folium.CircleMarker(
        [lat, lon], radius=9, color="#0f6e56", fill=True, fill_color="#0f6e56", tooltip="This home"
    ).add_to(fmap)
    for row in comps.itertuples(index=False):
        folium.CircleMarker(
            [row.latitude, row.longitude],
            radius=5,
            color="#a33b24",
            fill=True,
            fill_color="#a33b24",
            tooltip=f"{_money(row.price_clean)} · {row.km:.1f} km",
        ).add_to(fmap)
    out = st_folium(fmap, height=460, returned_objects=["last_clicked"], key=f"map-{city}")
    click = (out or {}).get("last_clicked")
    if click:
        token = (round(float(click["lat"]), 5), round(float(click["lng"]), 5))
        if token != st.session_state.get("click_token"):
            st.session_state.click_token = token
            st.session_state.lat = token[0]
            st.session_state.lon = token[1]
            st.rerun()
    if len(comps) == 0:
        st.caption(
            "No local listing file yet, so there are no nearby ads to show. The estimate still runs."
        )
    else:
        st.caption(
            "Coral dots are nearby ads from the cleaned snapshot. They are not model inputs."
        )


def _stay(scored: dict) -> None:
    nights = scored["booked_nights"]
    revenue = scored["annual_revenue"]
    rough = " Rough estimate." if scored["nights_rough_estimate"] else ""
    area = scored.get("area_median_nights")
    area_bit = ""
    if area is not None and scored.get("area_name"):
        # Austin's cleaned neighbourhood field is a ZIP code. Naming it
        # keeps "78701" from reading like a stray number.
        place = _place_name(scored["area_name"])
        area_bit = f" The usual nights figure for {place} is about {area:.0f}."
    elif area is not None:
        area_bit = f" The citywide usual nights figure is about {area:.0f}."
    yield_bit = ""
    if scored.get("gross_yield") is not None:
        yield_bit = (
            f" Rough gross yield {scored['gross_yield']:.1%} on the purchase price you entered. "
            "That is predicted revenue divided by that price, before mortgage, taxes, fees, and repairs."
        )
    note = (
        f"On held-out blocks the average miss was {scored['nights_holdout_mae']:.0f} nights "
        f"(R² {scored['nights_holdout_r2']:.2f}), against {scored['nights_baseline_mae']:.0f} "
        "for the usual nights in the area. "
    )
    if scored["nights_rough_estimate"]:
        note += (
            "That missed a bar set before the test, so this number stays a rough estimate. "
            "The label itself comes from review counts, not from a calendar of bookings."
        )
    if nights < 0.5:
        # A linear nights model can hit the floor on an unusual home, such as
        # a 10-bedroom shared room. Zero would read as a forecast. It is the floor.
        note += (
            " This guess landed at the bottom of the scale. Do not read zero nights "
            "as a forecast that the home would never book."
        )
    st.markdown(
        '<div class="stay-grid">'
        '<div class="card"><p class="eyebrow">Booked nights</p>'
        f'<p class="price" style="font-size:2.4rem">{nights:.0f}<span>nights / year</span></p>'
        f'<p class="muted">{rough.strip() or "Review-based estimate."}{area_bit}</p></div>'
        '<div class="card"><p class="eyebrow">Estimated yearly revenue</p>'
        f'<p class="price" style="font-size:2.4rem">{_money(revenue)}</p>'
        '<p class="muted">Predicted rate times predicted nights. Not what a host would clear.</p></div>'
        '<div class="card"><p class="eyebrow">Grade in this city</p>'
        f'<p class="price" style="font-size:2.4rem">{scored["grade"]}'
        f'<span>{scored["grade_words"]}</span></p>'
        '<p class="muted">Where that gross sits among the training homes. Not a profit grade.</p></div>'
        "</div>"
        f'<p class="muted">{note}{yield_bit}</p>',
        unsafe_allow_html=True,
    )


def _result(scored: dict, comps: pd.DataFrame) -> None:
    rate = scored["nightly_rate"]
    low, high = scored["interval_low"], scored["interval_high"]
    coverage = scored["interval_coverage"]
    left, right = st.columns([1.08, 0.92], gap="large")
    with left:
        reasons = (
            "".join(
                f'<p class="reason"><strong>{item["label"]}</strong>. {item["sentence"]}</p>'
                for item in scored["reasons"]
            )
            or '<p class="muted">No single filled-in fact moved this estimate by a dollar or more.</p>'
        )
        blank = ""
        if scored["blank_fields"]:
            blank = (
                " Fields left blank, including amenities and reviews when you did not fill them in, "
                "use a typical training value rather than zero."
            )
        st.markdown(
            '<div class="card"><p class="eyebrow">Estimated nightly rate</p>'
            f'<p class="price">{_money(rate)}<span>per night</span></p>'
            f'<p class="muted">Range {_money(low)} to {_money(high)}. '
            f"On held-out blocks this range covered {coverage:.0%} of homes. "
            "That is the measured coverage, not a promise for this address.</p>"
            f'<p class="muted">The rate model is {scored["library"].replace("_", " ")}.</p>'
            "</div>"
            '<div class="card" style="margin-top:0.9rem"><p class="eyebrow">Why this rate</p>'
            f"{reasons}"
            '<p class="muted">Each line puts one fact back to the typical training value and '
            f"reads the dollar change. It was not used to pick the model.{blank}</p></div>",
            unsafe_allow_html=True,
        )
    with right:
        if len(comps) == 0:
            body = '<p class="muted">Download the city snapshot to see nearby ads.</p>'
        else:
            prices = [float(row.price_clean) for row in comps.itertuples(index=False)]
            median = float(pd.Series(prices).median())
            rows = []
            for row in comps.itertuples(index=False):
                neighbourhood = _place_name(str(row.neighbourhood_cleansed).replace("<", ""))
                rows.append(
                    f'<div class="comp"><div><strong>{_money(row.price_clean)}</strong>'
                    f'<div class="muted">{neighbourhood}</div></div>'
                    f'<div style="text-align:right"><span class="pill">{row.km:.1f} km</span>'
                    f'<div class="muted">{int(row.bedrooms)} bd</div></div></div>'
                )
            body = (
                f'<p class="muted">Same room type when there are enough, within one bedroom. '
                f"Median of these five: {_money(median)}. The estimate is {_money(rate)}.</p>"
                + "".join(rows)
            )
        st.markdown(
            '<div class="card"><p class="eyebrow">Nearby advertised rates</p>' + body + "</div>",
            unsafe_allow_html=True,
        )


def _results_page() -> None:
    payload = _study()
    st.markdown('<p class="eyebrow">Primary holdout · scored once</p>', unsafe_allow_html=True)
    st.markdown('<p class="wordmark">What the model actually did</p>', unsafe_allow_html=True)
    if not payload:
        st.info("No study report yet. From the repo root: python -m evaluation.run_study")
        return
    st.markdown(payload["selection_reason"])
    st.markdown(payload["challenger_reason"])
    rows = []
    shipped = payload["shipped_model"]
    for city in payload["cities"]:
        block = payload["cities"][city]
        hold = block["price"]["holdout"][shipped]
        boot = block["price"]["bootstrap"]
        rows.append(
            {
                "City": block["display"],
                "R²": f"{hold['r2']:.3f}",
                "R² interval": f"{boot['r2']['low']:.3f} to {boot['r2']['high']:.3f}",
                "MAE": f"${hold['mae']:,.2f}",
                "MAE interval": f"${boot['mae']['low']:,.2f} to ${boot['mae']['high']:,.2f}",
                "Median % error": f"{hold['median_ape']:.1f}%",
            }
        )
    st.table(pd.DataFrame(rows))
    st.markdown(payload["nights_reason"])
    st.markdown(payload["nights_selection_reason"])
    night_rows = []
    nights_name = payload["nights_shipped_model"]
    for city in payload["cities"]:
        block = payload["cities"][city]
        hold = block["nights"]["holdout"][nights_name]
        night_rows.append(
            {
                "City": block["display"],
                "R²": f"{hold['r2']:.3f}",
                "MAE nights": f"{hold['mae']:.1f}",
                "Usual nights MAE": f"{block['nights']['holdout']['area_median']['mae']:.1f}",
                "Cleared the bar": "Yes" if block["nights"]["cleared_bar"] else "No",
            }
        )
    st.table(pd.DataFrame(night_rows))
    st.caption(
        "The other seeds were not used to pick the model. "
        "San Diego's nightly-rate R² moves a lot across those seeds. "
        "The headline numbers are the primary holdout."
    )


def main() -> None:
    page = st.sidebar.radio(
        "Study", ["Estimate a home", "Holdout results"], label_visibility="collapsed"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "A personal project on advertised rates in Austin, San Diego, and Seattle. "
        "It runs on this machine only."
    )
    if page == "Estimate a home":
        _predict_page()
    else:
        _results_page()


main()
