"""Synthetic calendars for the booked-night rules.

Each test builds the snapshots a daily collector would have stored and checks
the label. The cutoffs are the ones documented in docs/CALENDAR.md.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from data.calendar_labels import (
    BOOKED,
    CENSORED,
    OWNER_BLOCK,
    TOO_SHORT,
    UNBOOKED,
    UNCERTAIN,
    UNRESOLVED,
    WEEKS_BEFORE_RETRAIN,
    Rules,
    label_frame,
)
from data.calendar_sample import SAMPLE_PATH, allocate_strata, build_sample_document, draw_ids

ROOT = Path(__file__).resolve().parents[1]


def _rows(listing: str, snapshots: list[tuple[date, dict[date, tuple[bool, int]]]]) -> pd.DataFrame:
    records = []
    for fetch, days in snapshots:
        for night, (available, min_nights) in sorted(days.items()):
            records.append(
                {
                    "listing_id": listing,
                    "city": "austin",
                    "date": night.isoformat(),
                    "available": int(available),
                    "min_nights": min_nights,
                    "price": "",
                    "fetched_at": f"{fetch.isoformat()}T12:00:00Z",
                }
            )
    return pd.DataFrame(records)


def _labels(frame: pd.DataFrame, rules: Rules | None = None) -> tuple[pd.DataFrame, pd.Series]:
    nights, summary = label_frame(frame, rules)
    by_date = nights.set_index("date")["label"]
    return nights, by_date


def _summary(frame: pd.DataFrame, rules: Rules | None = None) -> pd.Series:
    _nights, summary = label_frame(frame, rules)
    return summary.iloc[0]


def test_short_near_flip_is_a_booking():
    stay = [date(2026, 4, 10), date(2026, 4, 11), date(2026, 4, 12)]
    open_days = {night: (True, 2) for night in stay}
    closed_days = {night: (False, 2) for night in stay}
    frame = _rows(
        "1",
        [
            (date(2026, 4, 1), open_days),
            (date(2026, 4, 3), closed_days),
            (date(2026, 4, 12), closed_days),
        ],
    )
    _nights, labels = _labels(frame)
    assert list(labels.loc[[night.isoformat() for night in stay]]) == [BOOKED, BOOKED, BOOKED]
    summary = _summary(frame)
    assert summary["booked_nights"] == 3
    assert summary["booked_rate"] == 1.0


def test_long_block_closed_far_ahead_is_an_owner_block():
    start = date(2026, 6, 1)
    nights = [start + timedelta(days=offset) for offset in range(21)]
    opened = {night: (True, 1) for night in nights}
    closed = {night: (False, 1) for night in nights}
    rules = Rules(resolve_slack_days=180)
    frame = _rows(
        "1",
        [
            (date(2026, 1, 1), opened),
            (date(2026, 1, 2), closed),
            (date(2026, 6, 21), closed),
        ],
    )
    _nights, labels = _labels(frame, rules)
    assert set(labels) == {OWNER_BLOCK}
    summary = _summary(frame, rules)
    assert summary["booked_nights"] == 0
    assert summary["owner_block_nights"] == 21
    assert pd.isna(summary["booked_rate"])


def test_already_closed_near_the_date_is_censored():
    stay = [date(2026, 5, 10), date(2026, 5, 11), date(2026, 5, 12)]
    days = {night: (False, 1) for night in stay}
    days[date(2026, 5, 13)] = (True, 1)
    frame = _rows("1", [(date(2026, 5, 5), days), (date(2026, 5, 12), days)])
    _nights, labels = _labels(frame)
    assert list(labels.loc[[night.isoformat() for night in stay]]) == [CENSORED, CENSORED, CENSORED]
    assert _summary(frame)["booked_nights"] == 0


def test_reopened_night_is_not_booked():
    stay = [date(2026, 4, 10), date(2026, 4, 11), date(2026, 4, 12)]
    opened = {night: (True, 1) for night in stay}
    closed = {night: (False, 1) for night in stay}
    frame = _rows(
        "1",
        [
            (date(2026, 4, 1), opened),
            (date(2026, 4, 3), closed),
            (date(2026, 4, 9), opened),
            (date(2026, 4, 12), opened),
        ],
    )
    _nights, labels = _labels(frame)
    assert set(labels) == {UNBOOKED}
    assert _summary(frame)["booked_nights"] == 0
    assert _summary(frame)["unbooked_nights"] == 3


def test_far_tail_through_the_end_of_the_calendar_is_a_closed_window():
    fetch = date(2026, 7, 1)
    nights = [fetch + timedelta(days=offset) for offset in range(40)]
    days = {night: (True, 1) for night in nights[:10]}
    days.update({night: (False, 1) for night in nights[10:]})
    # The tail nights are still in the future relative to this one fetch, so
    # shift the fetch to the last night. The tail is then elapsed and excluded.
    last = nights[-1]
    frame = _rows("1", [(last, days)])
    summary = _summary(frame, Rules(resolve_slack_days=40))
    assert summary["closed_window_nights"] == 30
    assert summary["unbooked_nights"] == 10
    assert summary["booked_nights"] == 0
    assert summary["booked_rate"] == 0.0


def test_one_night_shorter_than_min_nights_is_not_a_stay():
    night = date(2026, 4, 20)
    neighbors = [date(2026, 4, 19), night, date(2026, 4, 21)]
    opened = {day: (True, 3) for day in neighbors}
    closed = {date(2026, 4, 19): (True, 3), night: (False, 3), date(2026, 4, 21): (True, 3)}
    frame = _rows(
        "1",
        [
            (date(2026, 4, 10), opened),
            (date(2026, 4, 12), closed),
            (date(2026, 4, 21), closed),
        ],
    )
    _nights, labels = _labels(frame)
    assert labels.loc[night.isoformat()] == TOO_SHORT
    summary = _summary(frame)
    assert summary["booked_nights"] == 0
    assert summary["too_short_nights"] == 1
    assert summary["unbooked_nights"] == 2
    assert summary["booked_rate"] == 0.0


def test_booked_rate_ignores_blocks_and_uses_only_offered_nights():
    rules = Rules(resolve_slack_days=120)
    base = date(2026, 3, 1)
    offered = [base + timedelta(days=offset) for offset in range(40)]
    booked = offered[10:20]
    blocked = [date(2026, 1, 1) + timedelta(days=offset) for offset in range(14)]
    first = {night: (True, 1) for night in offered}
    second = {night: (night not in booked, 1) for night in offered}
    confirm = dict(second)
    blocked_days = {night: (False, 1) for night in blocked}
    frame = _rows(
        "1",
        [
            (date(2025, 9, 1), blocked_days),
            (date(2026, 2, 20), first),
            (date(2026, 2, 21), second),
            (offered[-1], {**confirm, **blocked_days}),
        ],
    )
    summary = _summary(frame, rules)
    assert summary["booked_nights"] == 10
    assert summary["unbooked_nights"] == 30
    assert summary["owner_block_nights"] == 14
    assert summary["offered_nights"] == 40
    assert abs(summary["booked_rate"] - 0.25) < 1e-9


def test_future_flip_is_not_in_the_rate():
    stay = [date(2026, 12, 1), date(2026, 12, 2), date(2026, 12, 3)]
    opened = {night: (True, 1) for night in stay}
    closed = {night: (False, 1) for night in stay}
    frame = _rows("1", [(date(2026, 11, 1), opened), (date(2026, 11, 2), closed)])
    summary = _summary(frame)
    assert summary["booked_nights"] == 0
    assert summary["future_booked_nights"] == 3
    assert pd.isna(summary["booked_rate"])


def test_a_stale_look_does_not_resolve_the_night():
    night = date(2026, 3, 20)
    frame = _rows(
        "1",
        [
            (date(2026, 3, 1), {night: (True, 1)}),
            (date(2026, 4, 10), {date(2026, 4, 10): (True, 1)}),
        ],
    )
    _nights, labels = _labels(frame)
    assert labels.loc[night.isoformat()] == UNRESOLVED


def test_medium_lead_and_a_long_close_in_block_stay_uncertain():
    short = [date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3)]
    long = [date(2026, 6, 10) + timedelta(days=offset) for offset in range(16)]
    # Extra nights stay open so closing the long run is not a mass block
    # of the whole calendar. Sixteen nights is 100% of the open calendar
    # when they are the only nights, and the mass rule would call that a block.
    held = [date(2026, 6, 26) + timedelta(days=offset) for offset in range(20)]
    rules = Rules(resolve_slack_days=40)
    frame = _rows(
        "1",
        [
            (date(2026, 3, 13), {night: (True, 1) for night in short}),
            (date(2026, 4, 12), {night: (False, 1) for night in short}),
            (date(2026, 6, 1), {night: (True, 1) for night in long + held}),
            (date(2026, 6, 2), {night: ((night not in long), 1) for night in long + held}),
            (
                long[-1],
                {night: (False, 1) for night in short + long}
                | {night: (True, 1) for night in held},
            ),
        ],
    )
    _nights, labels = _labels(frame, rules)
    assert set(labels.loc[[night.isoformat() for night in short]]) == {UNCERTAIN}
    assert set(labels.loc[[night.isoformat() for night in long]]) == {UNCERTAIN}
    assert _summary(frame, rules)["booked_nights"] == 0


def test_a_wide_gap_does_not_turn_a_short_near_flip_into_a_mass_block():
    stay = [date(2026, 5, 20), date(2026, 5, 21), date(2026, 5, 22)]
    opened = {night: (True, 1) for night in stay}
    closed = {night: (False, 1) for night in stay}
    frame = _rows(
        "1",
        [
            (date(2026, 4, 10), opened),
            (date(2026, 5, 10), closed),
            (date(2026, 5, 22), closed),
        ],
    )
    _nights, labels = _labels(frame)
    assert set(labels) == {BOOKED}


def test_half_the_open_calendar_closing_overnight_is_a_block():
    rules = Rules(resolve_slack_days=40)
    nights = [date(2026, 5, 1) + timedelta(days=offset) for offset in range(30)]
    opened = {night: (True, 1) for night in nights}
    closed_nights = set(nights[:20])
    closed = {night: (night not in closed_nights, 1) for night in nights}
    frame = _rows(
        "1",
        [
            (date(2026, 4, 20), opened),
            (date(2026, 4, 21), closed),
            (nights[-1], closed),
        ],
    )
    summary = _summary(frame, rules)
    assert summary["owner_block_nights"] == 20
    assert summary["booked_nights"] == 0
    assert summary["unbooked_nights"] == 10


def test_draw_is_stable_and_the_committed_panel_matches_it():
    frame = pd.DataFrame(
        {
            "listing_id": [str(item) for item in range(20)],
            "stratum": ["a|entire"] * 10 + ["b|private"] * 6 + ["c|shared"] * 4,
        }
    )
    same = draw_ids(frame, 8, 17, 0)
    assert same == draw_ids(frame.iloc[::-1], 8, 17, 0)
    assert len(same) == 8
    alloc = allocate_strata({"a|entire": 10, "b|private": 6, "c|shared": 4}, 8)
    assert sum(alloc.values()) == 8
    assert all(alloc[key] <= {"a|entire": 10, "b|private": 6, "c|shared": 4}[key] for key in alloc)

    document = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    fresh = build_sample_document()
    assert document["seed"] == fresh["seed"] == 17
    assert document["per_city"] == 300
    for city in ("austin", "san-diego", "seattle"):
        assert document["cities"][city]["listing_ids"] == fresh["cities"][city]["listing_ids"]
        assert len(document["cities"][city]["listing_ids"]) == min(
            300, fresh["cities"][city]["source_rows"]
        )


def test_docs_state_the_wait_and_the_branch():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    findings = (ROOT / "reports" / "FINDINGS.md").read_text(encoding="utf-8")
    calendar = (ROOT / "docs" / "CALENDAR.md").read_text(encoding="utf-8")
    assert WEEKS_BEFORE_RETRAIN == 26
    for text in (readme, findings, calendar):
        assert "26 weeks" in text
        assert "calendar-data" in text
        assert "45 days" in text
        assert "90 days" in text
    for text in (readme, findings):
        assert "fresh spatial holdout" in text
