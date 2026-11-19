"""Infer booked nights from repeated calendar snapshots.

A public Airbnb calendar says whether a night is available. It does not say
why a night is unavailable. The rules below are the open version of the
distinction AirDNA makes with a private model, and of the rule in the daily
scrape literature: count a booking only when a night was seen available and
then closed. They are written so a test can replay them on a synthetic calendar.

The rate is the share of resolved, offered nights that ended booked. Blocked
nights, the closed far edge of a host's calendar, and nights we never saw
open are counted beside that rate and left out of it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

# Seen still open inside this many days, then closed: close to the date.
# A short urban trip usually books inside about a month. Forty-five days
# leaves room for a missed scrape without calling a six-month closure "close".
NEAR_LEAD_DAYS = 45

# Already closed when the check-in was still this far out: far ahead.
# Ninety days is Airbnb's shortest standard availability window (3 months;
# the other settings are 6, 9, 12, and 24). A closure that is in place at
# that distance was made while the night was still in host-planning range.
FAR_LEAD_DAYS = 90

# A fortnight. Inside Airbnb's fallback stay is 3 nights, and the San
# Francisco figure they use when a city has published one is 5.5. Fourteen
# nights is past a short stay.
LONG_BLOCK_NIGHTS = 14

# One night shorter than a long block, so the two labels do not overlap.
MAX_BOOKING_NIGHTS = LONG_BLOCK_NIGHTS - 1

# Unavailable from some night through the last night the snapshot publishes.
# A short tail is not enough to call the availability window shut.
CLOSED_WINDOW_NIGHTS = 21

# One snapshot closes at least this share of the nights that were open, and
# at least a long block of them, and the two snapshots are at most this many
# days apart. A guest does not reserve half an open calendar overnight.
# A month of real bookings can, so the rule stays off when the gap is wider.
MASS_BLOCK_FRACTION = 0.50
MASS_BLOCK_MIN_NIGHTS = LONG_BLOCK_NIGHTS
MASS_BLOCK_MAX_GAP_DAYS = 3

# An elapsed night counts only when some snapshot falls this close to it.
# Daily collection survives one missed day. A monthly file does not, which
# is why the monthly fallback cannot fill in a booked-night rate.
RESOLVE_SLACK_DAYS = 2

# Half a year of daily snapshots before the first retrain. The reasoning is
# in docs/CALENDAR.md: about 13 weeks until nights we first saw far out have
# actually occurred, then about 12 weeks to accumulate ~80 offered nights.
WEEKS_BEFORE_RETRAIN = 26

BOOKED = "booked"
UNBOOKED = "unbooked"
OWNER_BLOCK = "owner_block"
CLOSED_WINDOW = "closed_window"
CENSORED = "censored"
UNCERTAIN = "uncertain"
TOO_SHORT = "too_short"
UNRESOLVED = "unresolved"
STILL_OPEN = "still_open"

ELAPSED_LABELS = (
    BOOKED,
    UNBOOKED,
    OWNER_BLOCK,
    CLOSED_WINDOW,
    CENSORED,
    UNCERTAIN,
    TOO_SHORT,
    UNRESOLVED,
)
# Nights in the denominator of the offered booked-night rate.
OFFERED_LABELS = (BOOKED, UNBOOKED)


@dataclass(frozen=True)
class Rules:
    near_lead_days: int = NEAR_LEAD_DAYS
    far_lead_days: int = FAR_LEAD_DAYS
    long_block_nights: int = LONG_BLOCK_NIGHTS
    max_booking_nights: int = MAX_BOOKING_NIGHTS
    closed_window_nights: int = CLOSED_WINDOW_NIGHTS
    mass_block_fraction: float = MASS_BLOCK_FRACTION
    mass_block_min_nights: int = MASS_BLOCK_MIN_NIGHTS
    mass_block_max_gap_days: int = MASS_BLOCK_MAX_GAP_DAYS
    resolve_slack_days: int = RESOLVE_SLACK_DAYS


@dataclass(frozen=True)
class Obs:
    available: bool
    min_nights: int | None


def contiguous_runs(nights: Iterable[date]) -> list[list[date]]:
    ordered = sorted(nights)
    if not ordered:
        return []
    runs = [[ordered[0]]]
    for night in ordered[1:]:
        if night - runs[-1][-1] == timedelta(days=1):
            runs[-1].append(night)
        else:
            runs.append([night])
    return runs


def _unavailable_tail(days: dict[date, Obs]) -> list[date]:
    if not days:
        return []
    cursor = max(days)
    tail: list[date] = []
    while cursor in days and not days[cursor].available:
        tail.append(cursor)
        cursor -= timedelta(days=1)
    tail.reverse()
    return tail


def _set_state(
    history: dict[date, list[tuple[date, str]]], night: date, fetch: date, state: str
) -> None:
    series = history.setdefault(night, [])
    if series and series[-1][1] == state and series[-1][0] == fetch:
        return
    series.append((fetch, state))


def _label_first_seen(
    history: dict[date, list[tuple[date, str]]],
    days: dict[date, Obs],
    fetch: date,
    new_nights: list[date],
    rules: Rules,
) -> None:
    tail = _unavailable_tail(days)
    tail_set = set(tail) if len(tail) >= rules.closed_window_nights else set()
    plain_unavailable: list[date] = []
    for night in new_nights:
        obs = days[night]
        if obs.available:
            _set_state(history, night, fetch, STILL_OPEN)
        elif night in tail_set:
            _set_state(history, night, fetch, CLOSED_WINDOW)
        else:
            plain_unavailable.append(night)
    for run in contiguous_runs(plain_unavailable):
        lead = (run[0] - fetch).days
        if len(run) >= rules.long_block_nights and lead >= rules.far_lead_days:
            state = OWNER_BLOCK
        else:
            state = CENSORED
        for night in run:
            _set_state(history, night, fetch, state)


def _classify_run(
    run: list[date],
    previous: dict[date, Obs],
    prev_fetch: date,
    fetch: date,
    rules: Rules,
) -> str:
    length = len(run)
    first = run[0]
    lead_lo = (first - fetch).days
    lead_hi = (first - prev_fetch).days
    posted = previous[first].min_nights
    min_nights = 1 if posted is None or posted < 1 else int(posted)
    if length >= rules.long_block_nights and lead_lo >= rules.far_lead_days:
        return OWNER_BLOCK
    if length > rules.max_booking_nights:
        return UNCERTAIN
    if lead_hi > rules.near_lead_days:
        return UNCERTAIN
    if length < min_nights:
        return TOO_SHORT
    return BOOKED


def _apply_flips(
    history: dict[date, list[tuple[date, str]]],
    previous: dict[date, Obs],
    current: dict[date, Obs],
    prev_fetch: date,
    fetch: date,
    rules: Rules,
) -> None:
    both = [night for night in current if night in previous]
    flipped: list[date] = []
    for night in both:
        was_open = previous[night].available
        now_open = current[night].available
        if was_open and not now_open:
            flipped.append(night)
        elif not was_open and now_open:
            _set_state(history, night, fetch, STILL_OPEN)
    if not flipped:
        return
    gap = (fetch - prev_fetch).days
    open_before = sum(1 for night in both if previous[night].available)
    mass = (
        gap <= rules.mass_block_max_gap_days
        and len(flipped) >= rules.mass_block_min_nights
        and open_before > 0
        and (len(flipped) / open_before) >= rules.mass_block_fraction
    )
    if mass:
        for night in flipped:
            _set_state(history, night, fetch, OWNER_BLOCK)
        return
    for run in contiguous_runs(flipped):
        state = _classify_run(run, previous, prev_fetch, fetch, rules)
        for night in run:
            _set_state(history, night, fetch, state)


def simulate(
    snapshots: dict[date, dict[date, Obs]],
    rules: Rules | None = None,
) -> dict[date, list[tuple[date, str]]]:
    """State after each fetch. The keys are nights. Values are (fetch, state)."""
    rules = rules or Rules()
    history: dict[date, list[tuple[date, str]]] = {}
    fetches = sorted(snapshots)
    for index, fetch in enumerate(fetches):
        current = snapshots[fetch]
        new_nights = [night for night in current if night not in history]
        if new_nights:
            _label_first_seen(history, current, fetch, new_nights, rules)
        if index == 0:
            continue
        _apply_flips(
            history, snapshots[fetches[index - 1]], current, fetches[index - 1], fetch, rules
        )
    return history


def _state_as_of(series: list[tuple[date, str]], fetch: date) -> str:
    state = STILL_OPEN
    for seen_at, seen_state in series:
        if seen_at <= fetch:
            state = seen_state
        else:
            break
    return state


def resolve_nights(
    snapshots: dict[date, dict[date, Obs]],
    history: dict[date, list[tuple[date, str]]],
    rules: Rules | None = None,
) -> list[dict]:
    """One row per night. Future nights keep their provisional state."""
    rules = rules or Rules()
    fetches = sorted(snapshots)
    if not fetches:
        return []
    last_fetch = fetches[-1]
    slack = timedelta(days=rules.resolve_slack_days)
    rows = []
    for night in sorted(history):
        elapsed = night <= last_fetch
        resolve_fetch = None
        for fetch in fetches:
            if fetch <= night + slack and night in snapshots[fetch]:
                resolve_fetch = fetch
        if not elapsed:
            rows.append({"date": night, "label": history[night][-1][1], "elapsed": False})
            continue
        stale = resolve_fetch is None or resolve_fetch < night - slack
        if stale or resolve_fetch is None:
            rows.append({"date": night, "label": UNRESOLVED, "elapsed": True})
            continue
        if snapshots[resolve_fetch][night].available:
            label = UNBOOKED
        else:
            label = _state_as_of(history[night], resolve_fetch)
            if label == STILL_OPEN:
                label = CENSORED
        rows.append({"date": night, "label": label, "elapsed": True})
    return rows


def summarize_nights(listing_id: str, rows: list[dict], snapshots: int, city: str = "") -> dict:
    elapsed = [row for row in rows if row["elapsed"]]
    future = [row for row in rows if not row["elapsed"]]
    counts = {label: 0 for label in ELAPSED_LABELS}
    for row in elapsed:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    offered = counts[BOOKED] + counts[UNBOOKED]
    elapsed_dates = [row["date"] for row in elapsed]
    summary = {
        "listing_id": str(listing_id),
        "city": city,
        "snapshots": int(snapshots),
        "window_start": min(elapsed_dates).isoformat() if elapsed_dates else "",
        "window_end": max(elapsed_dates).isoformat() if elapsed_dates else "",
        "offered_nights": offered,
        "booked_rate": (counts[BOOKED] / offered) if offered else None,
    }
    for label in ELAPSED_LABELS:
        summary[f"{label}_nights"] = counts[label]
    summary["future_booked_nights"] = sum(1 for row in future if row["label"] == BOOKED)
    summary["future_owner_block_nights"] = sum(1 for row in future if row["label"] == OWNER_BLOCK)
    summary["future_closed_window_nights"] = sum(
        1 for row in future if row["label"] == CLOSED_WINDOW
    )
    return summary


def _utc_date(value: object) -> date:
    if isinstance(value, datetime):
        stamp = pd.Timestamp(value)
    elif isinstance(value, date):
        return value
    else:
        stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"Cannot read a date from {value!r}")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.date()


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"t", "true", "1", "yes"}:
        return True
    if text in {"f", "false", "0", "no"}:
        return False
    raise ValueError(f"Cannot read availability from {value!r}")


def _as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    return int(float(text))


def snapshots_from_frame(part: pd.DataFrame) -> dict[date, dict[date, Obs]]:
    """Collapse rows to one observation per fetch-day and night. Later wins."""
    ordered = part.sort_values("fetched_at")
    snapshots: dict[date, dict[date, Obs]] = {}
    for row in ordered.itertuples(index=False):
        fetch = _utc_date(row.fetched_at)
        night = _utc_date(row.date)
        snapshots.setdefault(fetch, {})[night] = Obs(
            _as_bool(row.available),
            _as_optional_int(row.min_nights),
        )
    return snapshots


def label_frame(
    frame: pd.DataFrame, rules: Rules | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Night labels and one summary row per listing.

    ``frame`` uses the collector schema: listing_id, date, available,
    min_nights, price, fetched_at. ``city`` is optional. Price is stored by
    the collector and is not a rule. The public calendar often leaves it blank
    on both open and closed nights.
    """
    rules = rules or Rules()
    night_columns = ["listing_id", "city", "date", "label", "elapsed"]
    summary_columns = list(summarize_nights("", [], 0))
    if frame.empty:
        return pd.DataFrame(columns=night_columns), pd.DataFrame(columns=summary_columns)

    working = frame.copy()
    working["listing_id"] = working["listing_id"].map(_normalize_listing_id)
    if "city" not in working.columns:
        working["city"] = ""
    night_rows: list[dict] = []
    summaries: list[dict] = []
    for listing_id, part in working.groupby("listing_id", sort=True):
        city = str(part["city"].iloc[0]) if part["city"].notna().any() else ""
        snapshots = snapshots_from_frame(part)
        history = simulate(snapshots, rules)
        resolved = resolve_nights(snapshots, history, rules)
        for row in resolved:
            night_rows.append(
                {
                    "listing_id": listing_id,
                    "city": city,
                    "date": row["date"].isoformat(),
                    "label": row["label"],
                    "elapsed": bool(row["elapsed"]),
                }
            )
        summaries.append(summarize_nights(str(listing_id), resolved, len(snapshots), city))
    nights = pd.DataFrame(night_rows, columns=night_columns)
    summary = pd.DataFrame(summaries, columns=summary_columns)
    return nights, summary


def _normalize_listing_id(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def load_snapshot_files(paths: Iterable[Path | str]) -> pd.DataFrame:
    """Read collector gzip snapshots. The city is taken from the directory."""
    frames = []
    for path in paths:
        file_path = Path(path)
        frame = pd.read_csv(file_path, dtype={"listing_id": "string"})
        frame["city"] = _city_from_path(file_path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(
            columns=["listing_id", "date", "available", "min_nights", "price", "fetched_at", "city"]
        )
    return pd.concat(frames, ignore_index=True)


def _city_from_path(path: Path) -> str:
    parts = path.parts
    for marker in ("daily", "inside-airbnb"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts) - 1:
                return parts[index + 1]
    return ""
