"""Dollar drivers for one nightly-rate prediction.

Each driver puts one fact back to the training median and reads the change
in dollars. Derived fields stay consistent with the fact they come from:
bedrooms times guests is recomputed when either input moves, and the pin,
the distances, and the public-place score move together. A column that is
only a product of other columns is not given its own bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from features.columns import pretty

LOCATION = (
    "latitude",
    "longitude",
    "distance_to_center_km",
    "distance_to_airport_km",
    "attraction_index",
    "distance_to_coast_km",
)
ROOM = ("is_private_room", "is_shared_room", "is_hotel_room")
BUILDING = ("is_house", "is_condo")
# Not an input. It is bedrooms times guests, so it moves with those facts.
DERIVED = frozenset({"bedrooms_x_accommodates"})
_SIZE = frozenset({"bedrooms", "accommodates"})
_FLAGS = frozenset({"bath_is_shared", "is_superhost"})


def rate_reasons(model, row: pd.DataFrame, limit: int = 5) -> list[dict]:
    """Up to `limit` facts, largest dollar move first.

    ``model`` needs ``columns``, ``medians_``, and ``predict``. ``row`` is
    one home, with the same derived columns the model was trained on.
    """
    base = float(model.predict(row)[0])
    reasons: list[dict] = []
    for key, label, members, kind in _groups(list(model.columns)):
        if _already_typical(model, row, members):
            continue
        altered = _counterfactual(model, row, members, kind)
        delta = base - float(model.predict(altered)[0])
        if abs(delta) < 1.0:
            continue
        reasons.append(
            {
                "feature": key,
                "label": label,
                "dollars": float(delta),
                "sentence": _sentence(kind, members, row, model, delta),
                "columns": list(members),
            }
        )
    reasons.sort(key=lambda item: abs(item["dollars"]), reverse=True)
    return reasons[: int(limit)]


def _groups(columns: list[str]) -> list[tuple[str, str, list[str], str]]:
    used: set[str] = set()
    groups: list[tuple[str, str, list[str], str]] = []

    def take(key: str, label: str, members: tuple[str, ...], kind: str) -> None:
        present = [column for column in members if column in columns]
        if not present:
            return
        groups.append((key, label, present, kind))
        used.update(present)

    take("location", "Location", LOCATION, "location")
    take("room_type", "Room type", ROOM, "room")
    take("building", "Building", BUILDING, "building")
    for column in columns:
        if column in used or column in DERIVED:
            continue
        kind = "size" if column in _SIZE else "column"
        groups.append((column, pretty(column), [column], kind))
    return groups


def _already_typical(model, row: pd.DataFrame, members: list[str]) -> bool:
    for column in members:
        value = row.iloc[0][column]
        if not np.isfinite(value):
            continue
        if abs(float(value) - float(model.medians_[column])) >= 1e-9:
            return False
    return True


def _counterfactual(model, row: pd.DataFrame, members: list[str], kind: str) -> pd.DataFrame:
    altered = row.copy()
    for column in members:
        altered[column] = float(model.medians_[column])
    if kind == "size" and "bedrooms_x_accommodates" in model.columns:
        _sync_interaction(altered)
    return altered


def _sync_interaction(frame: pd.DataFrame) -> None:
    """Keep bedrooms × guests equal to the two inputs when both are known."""
    bedrooms = frame.iloc[0]["bedrooms"] if "bedrooms" in frame.columns else np.nan
    guests = frame.iloc[0]["accommodates"] if "accommodates" in frame.columns else np.nan
    if np.isfinite(bedrooms) and np.isfinite(guests):
        frame["bedrooms_x_accommodates"] = float(bedrooms) * float(guests)
    else:
        frame["bedrooms_x_accommodates"] = np.nan


def _sentence(kind: str, members: list[str], row: pd.DataFrame, model, delta: float) -> str:
    if kind == "location":
        fact = (
            "This pin is not the median training location. "
            "The coordinates, the distances, and the public-place score move together"
        )
    elif kind == "room":
        fact = f"The room type is {_room_name(row)}. The comparison puts those flags at the training median"
    elif kind == "building":
        fact = (
            f"The building is {_building_name(row)}. "
            "The comparison puts the house and condo flags at the training median"
        )
    else:
        column = members[0]
        fact = _column_fact(column, float(row.iloc[0][column]), float(model.medians_[column]))
    amount = f"${abs(delta):,.0f}"
    if delta > 0:
        move = f"Putting it back to that median lowers the nightly rate by {amount}."
    else:
        move = f"Putting it back to that median raises the nightly rate by {amount}."
    return f"{fact}. {move}"


def _column_fact(column: str, value: float, median: float) -> str:
    name = pretty(column)
    if column == "attraction_index":
        return (
            f"The public-place score is {_num(value)}. "
            f"The median training home's score is {_num(median)}"
        )
    if column.startswith(("is_", "has_")) or column in _FLAGS:
        now = "yes" if value >= 0.5 else "no"
        usual = "yes" if median >= 0.5 else "no"
        return f"{name} is {now}. The median training home is {usual}"
    return f"{name} is {_num(value)}. The median training home is {_num(median)}"


def _room_name(row: pd.DataFrame) -> str:
    if _on(row, "is_private_room"):
        return "a private room"
    if _on(row, "is_shared_room"):
        return "a shared room"
    if _on(row, "is_hotel_room"):
        return "a hotel room"
    return "an entire home"


def _building_name(row: pd.DataFrame) -> str:
    house = _on(row, "is_house")
    condo = _on(row, "is_condo")
    if house and not condo:
        return "a house"
    if condo and not house:
        return "a condo or apartment"
    if house and condo:
        return "marked as both a house and a condo"
    return "neither a house nor a condo"


def _on(row: pd.DataFrame, column: str) -> bool:
    if column not in row.columns:
        return False
    value = row.iloc[0][column]
    return bool(np.isfinite(value) and float(value) >= 0.5)


def _num(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return str(int(round(value)))
    return f"{value:.1f}"
