"""Local study app. Binds to 127.0.0.1 via .streamlit/config.toml.

    streamlit run app/streamlit_app.py

Pick a city, enter a home, and read the validated nightly rate first.
Nights booked, yearly revenue, and the letter grade sit below that, labeled
as rough estimates. This file does not deploy anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt
import folium
import pandas as pd
import requests
import streamlit as st
from app.locate import LocateError, check_pin, locate_address
from data.cities import CITIES, STUDY_CITIES
from models.artifacts import load_artifact
from models.serve import load_comp_frame, nearby_comps, score_home
from streamlit_folium import st_folium

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
EXAMPLES = {
    "Downtown Austin": ("austin", 30.2672, -97.7431),
    "Pacific Beach": ("san-diego", 32.797, -117.255),
    "Capitol Hill": ("seattle", 47.623, -122.319),
}
AMENITY_CHOICES = {
    "Air conditioning": "has_ac",
    "Parking": "has_parking",
    "Washer": "has_washer",
    "Kitchen": "has_kitchen",
    "Pool": "has_pool",
    "Hot tub": "has_hottub",
    "Waterfront": "has_waterfront",
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
      .muted { color: #6d6456; font-size: 0.92rem; line-height: 1.45; }
      h1, h2, h3 { font-family: Georgia, serif; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _fetch(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


@st.cache_resource(show_spinner=False)
def _model(city: str) -> dict:
    """Load one city's saved model once per process."""
    return load_artifact(city)


@st.cache_data(show_spinner=False)
def _comps_source(city: str) -> pd.DataFrame:
    return load_comp_frame(city)


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
        '<p class="muted">The nightly rate below is the validated estimate. '
        "Nights booked, yearly revenue, and the letter grade are rough estimates "
        "from a review-based nights model that missed its bar.</p>",
        unsafe_allow_html=True,
    )
    if "city" not in st.session_state:
        st.session_state.city = STUDY_CITIES[0]
        st.session_state.lat = float(CITIES[STUDY_CITIES[0]]["latitude"])
        st.session_state.lon = float(CITIES[STUDY_CITIES[0]]["longitude"])
    if "pending_pin" in st.session_state:
        pending_lat, pending_lon = st.session_state.pop("pending_pin")
        st.session_state.lat = pending_lat
        st.session_state.lon = pending_lon

    st.caption("Examples")
    buttons = st.columns(3)
    for column, (label, (city, lat, lon)) in zip(buttons, EXAMPLES.items(), strict=True):
        if column.button(label, use_container_width=True):
            st.session_state.city = city
            st.session_state.coords_city = city
            st.session_state.lat = lat
            st.session_state.lon = lon
            st.session_state.pop("scored", None)
            st.rerun()

    left, right = st.columns([0.92, 1.08], gap="large")
    with left:
        city = st.selectbox(
            "City",
            STUDY_CITIES,
            format_func=lambda key: CITIES[key]["display"],
            key="city",
        )
        if st.session_state.get("coords_city") != city:
            if st.session_state.get("coords_city") is not None:
                st.session_state.lat = float(CITIES[city]["latitude"])
                st.session_state.lon = float(CITIES[city]["longitude"])
                st.session_state.pop("scored", None)
            st.session_state.coords_city = city
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
                st.rerun()
        with st.form("home"):
            c1, c2, c3 = st.columns(3)
            bedrooms = c1.number_input("Bedrooms", min_value=0, max_value=10, value=2, step=1)
            bathrooms = c2.number_input(
                "Bathrooms", min_value=0.0, max_value=12.0, value=1.0, step=0.5
            )
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
            with st.expander("Advanced"):
                lat = st.number_input("Latitude", key="lat", format="%.5f")
                lon = st.number_input("Longitude", key="lon", format="%.5f")
            go = st.form_submit_button("Estimate this home", type="primary")

    pin_lat = float(st.session_state.lat)
    pin_lon = float(st.session_state.lon)
    comps = nearby_comps(
        _comps_source(st.session_state.city), pin_lat, pin_lon, room, float(bedrooms)
    )
    with right:
        _map(st.session_state.city, pin_lat, pin_lon, comps)

    if go:
        use_lat, use_lon = float(lat), float(lon)
        if address.strip():
            try:
                found = locate_address(address, city, _fetch)
            except LocateError as exc:
                st.warning(exc.message)
                use_lat, use_lon = float(lat), float(lon)
            else:
                use_lat, use_lon = found["latitude"], found["longitude"]
                st.caption(found["label"])
        try:
            check_pin(use_lat, use_lon, city)
            spec = {
                "latitude": use_lat,
                "longitude": use_lon,
                "bedrooms": int(bedrooms),
                "bathrooms": float(bathrooms),
                "accommodates": int(guests),
                "room_type": room,
                "purchase_price": home_price,
            }
            if building != "Not specified":
                spec["property_type"] = building
            spec.update(extras)
            shown = nearby_comps(_comps_source(city), use_lat, use_lon, room, float(bedrooms))
            if len(shown):
                spec["neighbourhood"] = str(shown.iloc[0]["neighbourhood_cleansed"])
            st.session_state.scored = score_home(city, spec, _model(city))
            st.session_state.scored_city = city
            st.session_state.shown_comps = shown
        except LocateError as exc:
            st.session_state.pop("scored", None)
            st.error(exc.message)
        except FileNotFoundError:
            st.error(
                "No saved model for this city. From the repo root: python -m evaluation.run_study"
            )

    if st.session_state.get("scored") and st.session_state.get("scored_city") == city:
        _rate(st.session_state.scored, st.session_state.get("shown_comps", comps))
        _rough(st.session_state.scored)
    else:
        st.markdown(
            '<p class="muted">Pick an example, type an address, or open Advanced and enter a '
            "latitude and longitude, then estimate. Blank fields use the training median, not zero.</p>",
            unsafe_allow_html=True,
        )


def _optional_inputs() -> dict:
    features: dict = {}
    with st.expander("If this home already has a listing"):
        st.caption(
            "Optional. Leave these alone for a home that has never been listed. "
            "A blank is the training median, not a zero."
        )
        if st.checkbox("I know the amenities"):
            picked = st.multiselect("Amenities", list(AMENITY_CHOICES))
            features["amenity_flags"] = {
                column: label in picked for label, column in AMENITY_CHOICES.items()
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
    out = st_folium(fmap, height=420, returned_objects=["last_clicked"], key=f"map-{city}")
    click = (out or {}).get("last_clicked")
    if click:
        token = (round(float(click["lat"]), 5), round(float(click["lng"]), 5))
        if token != st.session_state.get("click_token"):
            st.session_state.click_token = token
            st.session_state.pending_pin = token
            st.rerun()
    if len(comps) == 0:
        st.caption(
            "No local listing file yet, so there are no nearby ads to show. The estimate still runs."
        )
    else:
        st.caption(
            "Coral dots are nearby ads from the cleaned snapshot. They are not model inputs."
        )


def _rate(scored: dict, comps: pd.DataFrame) -> None:
    rate = scored["nightly_rate"]
    low, high = scored["interval_low"], scored["interval_high"]
    coverage = scored["interval_coverage"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Nightly rate", _money(rate))
    m2.metric("Low end of the range", _money(low))
    m3.metric("High end of the range", _money(high))
    st.caption(
        f"Ridge regression, a linear model. The band was built as an 80% range from training residuals. "
        f"On held-out blocks it covered {coverage:.0%} of homes. "
        "That is the measured coverage, not a promise for this address."
    )
    _comparison(rate, low, high, comps)
    _reasons(scored)


def _comparison(rate: float, low: float, high: float, comps: pd.DataFrame) -> None:
    st.markdown("#### This estimate and five nearby ads")
    rows = [{"Home": "This estimate", "Rate": rate, "Low": low, "High": high}]
    table_rows = []
    if len(comps) == 0:
        st.caption("Download the city snapshot to see nearby ads.")
    else:
        for row in comps.itertuples(index=False):
            place = _place_name(str(row.neighbourhood_cleansed).replace("<", ""))
            label = f"{place} · {row.km:.1f} km"
            price = float(row.price_clean)
            rows.append({"Home": label, "Rate": price, "Low": price, "High": price})
            table_rows.append(
                {
                    "Place": place,
                    "Advertised rate": _money(price),
                    "Bedrooms": int(row.bedrooms) if pd.notna(row.bedrooms) else None,
                    "Distance": f"{row.km:.1f} km",
                }
            )
    frame = pd.DataFrame(rows)
    bars = (
        alt.Chart(frame)
        .mark_bar(color="#0f6e56")
        .encode(
            y=alt.Y("Home:N", sort=None, title=None),
            x=alt.X("Rate:Q", title="Nightly rate ($)"),
            tooltip=["Home", alt.Tooltip("Rate:Q", format="$,.0f")],
        )
    )
    span = (
        alt.Chart(frame.iloc[[0]])
        .mark_rule(color="#1c1915", strokeWidth=2)
        .encode(y="Home:N", x="Low:Q", x2="High:Q")
    )
    st.altair_chart(bars + span, use_container_width=True)
    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)
        median = float(pd.Series([row["Rate"] for row in rows[1:]]).median())
        st.caption(
            f"The five ads have a median advertised rate of {_money(median)}. "
            f"The estimate is {_money(rate)}, and the black line is its range."
        )


def _reasons(scored: dict) -> None:
    reasons = scored["reasons"]
    st.markdown("#### Why this rate")
    st.caption(
        "Each bar is the dollar change from putting one filled-in fact back to the training median. "
        "Ridge regression, a linear model, is what produces the rate."
    )
    if not reasons:
        st.caption("No single filled-in fact moved this estimate by a dollar or more.")
        return
    frame = pd.DataFrame(
        {
            "Fact": [item["label"] for item in reasons],
            "Dollars": [item["dollars"] for item in reasons],
        }
    )
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            y=alt.Y("Fact:N", sort=None, title=None),
            x=alt.X("Dollars:Q", title="Change in the nightly rate ($)"),
            color=alt.condition(alt.datum.Dollars > 0, alt.value("#0f6e56"), alt.value("#a33b24")),
            tooltip=["Fact", alt.Tooltip("Dollars:Q", format="$,.0f")],
        )
    )
    st.altair_chart(chart, use_container_width=True)
    for item in reasons:
        st.markdown(f"**{item['label']}.** {item['sentence']}")
    if scored["blank_fields"]:
        st.caption(
            "Fields left blank, including amenities and reviews when you did not fill them in, "
            "use the training median rather than zero."
        )


def _rough(scored: dict) -> None:
    st.markdown("#### Rough estimates (not validated)")
    st.caption(
        "These three use the nights model. It did not clear a bar set before the test, "
        "and the nights label is built from reviews, not from a count of bookings. "
        "The letter grade ranks this rough yearly revenue inside the city. It is not a profit grade."
    )
    nights = scored["booked_nights"]
    revenue = scored["annual_revenue"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Nights booked", f"{nights:.0f}")
    c2.metric("Yearly revenue", _money(revenue))
    c3.metric("Grade", f"{scored['grade']} · {scored['grade_words']}")
    area = scored.get("area_median_nights")
    if area is not None and scored.get("area_name"):
        place = _place_name(scored["area_name"])
        st.caption(f"The median nights figure for {place} is about {area:.0f}.")
    elif area is not None:
        st.caption(f"The citywide median nights figure is about {area:.0f}.")
    note = (
        f"On held-out blocks the average miss was {scored['nights_holdout_mae']:.0f} nights "
        f"(R² {scored['nights_holdout_r2']:.2f}), against {scored['nights_baseline_mae']:.0f} "
        "for the local median nights. "
    )
    if scored["nights_rough_estimate"]:
        note += "That missed the bar, so this number stays a rough estimate."
    if nights < 0.5:
        note += (
            " This guess landed at the bottom of the scale. Do not read zero nights "
            "as a forecast that the home would never book."
        )
    if scored.get("gross_yield") is not None:
        note += (
            f" Rough gross yield {scored['gross_yield']:.1%} on the purchase price you entered. "
            "That is predicted revenue divided by that price, before mortgage, taxes, fees, and repairs."
        )
    st.caption(note)


def _within(y: pd.Series, pred: pd.Series, tol: float) -> float:
    return float((y.sub(pred).abs() / y.clip(lower=1e-9) <= tol).mean())


def _results_page() -> None:
    st.markdown('<p class="eyebrow">Primary holdout · scored once</p>', unsafe_allow_html=True)
    st.markdown('<p class="wordmark">What the model actually did</p>', unsafe_allow_html=True)
    adopted_path = REPORTS / "study_adopted.json"
    pred_path = REPORTS / "study_adopted_predictions.csv"
    if adopted_path.exists() and pred_path.exists():
        adopted = json.loads(adopted_path.read_text())
        predictions = pd.read_csv(pred_path)
        homes = adopted["overall"]["every_home_once"]
        st.metric(
            "Every home once, R²",
            f"{homes['r2']['point']:.3f}",
            help="Pooled holdout. Seattle's interval resamples 8 cells.",
        )
        share = _within(predictions["y_price"], predictions["price_ridge"], 0.20)
        base = _within(predictions["y_price"], predictions["price_area_median"], 0.20)
        rows = [
            {
                "Slice": "Every home once",
                "Homes": f"{int(homes['n']['point']):,}",
                "Within ±20%": f"{100 * share:.1f}%",
                "Local median within ±20%": f"{100 * base:.1f}%",
                "Median error": f"{homes['median_ape']['point']:.1f}%",
                "Less error": f"{100 * homes['error_removed']['point']:.1f}%",
            }
        ]
        labels = {
            "austin": "Austin, TX",
            "san-diego": "San Diego, CA",
            "seattle": "Seattle, WA",
        }
        for city, block in adopted["cities"].items():
            part = predictions[predictions["city"] == city]
            rows.append(
                {
                    "Slice": labels.get(city, city),
                    "Homes": f"{block['n']:,}",
                    "Within ±20%": f"{100 * _within(part['y_price'], part['price_ridge'], 0.20):.1f}%",
                    "Local median within ±20%": (
                        f"{100 * _within(part['y_price'], part['price_area_median'], 0.20):.1f}%"
                    ),
                    "Median error": f"{block['median_ape']:.1f}%",
                    "Less error": f"{100 * block['error_removed']:.1f}%",
                }
            )
        for name in (
            "pred_vs_actual.png",
            "error_by_tier.png",
            "ridge_coefficients.png",
            "model_comparison.png",
        ):
            path = FIGURES / name
            if path.exists():
                st.image(str(path))
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "Within ±20% is the share of held-out homes whose prediction is within 20% of the "
            "advertised rate. The local median column is the same share for the neighbourhood "
            "and room-type median. Seattle's bootstrap uses 8 held-out cells."
        )
    payload_path = REPORTS / "study_metrics.json"
    if not payload_path.exists():
        st.info("No study report yet. From the repo root: python -m evaluation.run_study")
        return
    payload = json.loads(payload_path.read_text())
    st.markdown("#### How ridge was chosen")
    st.markdown(payload["selection_reason"])
    st.caption(
        "That comparison used the columns before the added place and size features. "
        "The charts above are the model after those features were adopted."
    )
    st.markdown(payload["nights_reason"])


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
