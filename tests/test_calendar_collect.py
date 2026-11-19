"""The collector stops on a challenge and can fall back without touching the network."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from data.calendar_collect import (
    Decision,
    calendar_links_by_city,
    collect,
    decide_response,
    parse_calendar_payload,
    parse_price,
    write_snapshot,
)
from data.calendar_labels import load_snapshot_files

AUSTIN = "https://data.insideairbnb.com/united-states/tx/austin/2026-09-17/data/calendar.csv.gz"
SAN_DIEGO = (
    "https://data.insideairbnb.com/united-states/ca/san-diego/2026-09-20/data/calendar.csv.gz"
)
SEATTLE = "https://data.insideairbnb.com/united-states/wa/seattle/2026-09-20/data/calendar.csv.gz"
NOW = datetime(2026, 11, 19, 15, 0, tzinfo=UTC)


def _day(day: str, available: bool = True, min_nights: int = 2, price: str | None = None) -> dict:
    return {
        "calendarDate": day,
        "available": available,
        "minNights": min_nights,
        "price": {"localPriceFormatted": price},
    }


def test_parse_price_and_a_calendar_payload():
    assert parse_price("$1,250.00") == "1250"
    assert parse_price(None) == ""
    assert parse_price("0") == ""
    payload = {
        "data": {
            "merlin": {
                "pdpAvailabilityCalendar": {
                    "calendarMonths": [
                        {
                            "days": [
                                _day("2026-11-20", True, 2, None),
                                _day("2026-11-21", False, 1, "$80"),
                            ]
                        }
                    ]
                }
            }
        }
    }
    days = parse_calendar_payload(payload)
    assert days is not None
    assert days[0]["available"] == 1
    assert days[0]["price"] == ""
    assert days[1]["price"] == "80"
    body = json.dumps(payload).encode()
    decision = decide_response(200, "application/json", body)
    assert decision.kind == "ok"
    assert decide_response(403, "text/html", b"<html>captcha</html>").kind == "challenge"
    assert (
        decide_response(200, "text/html", b"<!doctype html><html>just a moment</html>").kind
        == "challenge"
    )
    assert decide_response(429, "application/json", b"{}").kind == "retry"
    assert decide_response(404, "application/json", b"{}").kind == "missing"
    missing = json.dumps({"errors": [{"message": "PersistedQueryNotFound"}]}).encode()
    assert decide_response(200, "application/json", missing).kind == "unusable"


def test_index_page_links_are_the_three_cities_only():
    html = f'<a href="{AUSTIN}"></a><a href="{SAN_DIEGO}"></a><a href="{SEATTLE}"></a>'
    html += '<a href="https://data.insideairbnb.com/united-states/ny/albany/2026-09-14/data/calendar.csv.gz"></a>'
    found = calendar_links_by_city(html)
    assert found["austin"] == [("2026-09-17", AUSTIN)]
    assert found["san-diego"][0][0] == "2026-09-20"
    assert found["seattle"][0][0] == "2026-09-20"
    assert "albany" not in found


def test_challenge_stops_before_the_next_listing(tmp_path: Path):
    calls: list[str] = []
    sleeps: list[float] = []

    def fetch(listing_id: str, _when: datetime) -> Decision:
        calls.append(listing_id)
        if listing_id == "1":
            return Decision(
                "ok", "ok", [{"date": "2026-11-20", "available": 1, "min_nights": 2, "price": ""}]
            )
        return Decision("challenge", "http 403")

    report = collect(
        {"austin": ["1", "2", "3"], "san-diego": ["9"]},
        tmp_path,
        now=NOW,
        delay=2,
        jitter=0,
        fetch=fetch,
        sleep=sleeps.append,
        fallback_html=lambda: "",
        download=lambda _url, _dest: None,
    )
    assert calls == ["1", "2"]
    assert sleeps == [2.0]
    assert report["mode"] == "airbnb_public"
    assert report["blocked"] is True
    assert report["cities"]["austin"]["succeeded"] == 1
    assert report["cities"]["san-diego"]["succeeded"] == 0
    saved = load_snapshot_files(tmp_path.rglob("*.csv.gz"))
    assert list(saved["listing_id"]) == ["1"]
    assert int(saved.iloc[0]["available"]) == 1


def test_cities_are_separated_by_the_same_pause(tmp_path: Path):
    sleeps: list[float] = []

    def fetch(_listing_id: str, _when: datetime) -> Decision:
        return Decision(
            "ok", "ok", [{"date": "2026-11-20", "available": 1, "min_nights": 1, "price": ""}]
        )

    collect(
        {"austin": ["1"], "san-diego": ["2"], "seattle": ["3"]},
        tmp_path,
        now=NOW,
        delay=2,
        jitter=0,
        fetch=fetch,
        sleep=sleeps.append,
    )
    assert sleeps == [2.0, 2.0]


def test_a_finished_full_panel_is_not_refetched_or_shrunk(tmp_path: Path):
    calls: list[str] = []

    def fetch(listing_id: str, _when: datetime) -> Decision:
        calls.append(listing_id)
        return Decision(
            "ok", "ok", [{"date": "2026-11-20", "available": 1, "min_nights": 1, "price": ""}]
        )

    collect(
        {"austin": ["1", "2"]},
        tmp_path,
        now=NOW,
        delay=0,
        jitter=0,
        fetch=fetch,
        sleep=lambda _seconds: None,
    )
    saved = list(tmp_path.rglob("*.csv.gz"))
    assert len(saved) == 1
    first_bytes = saved[0].read_bytes()
    again = collect(
        {"austin": ["1"]},
        tmp_path,
        now=NOW,
        delay=0,
        jitter=0,
        fetch=fetch,
        sleep=lambda _seconds: None,
        max_listings=1,
    )
    assert calls == ["1", "2"]
    assert again["cities"]["austin"]["notes"] == ["already collected today"]
    assert saved[0].read_bytes() == first_bytes


def test_a_capped_day_does_not_block_the_full_panel(tmp_path: Path):
    calls: list[str] = []

    def fetch(listing_id: str, _when: datetime) -> Decision:
        calls.append(listing_id)
        return Decision(
            "ok", "ok", [{"date": "2026-11-20", "available": 0, "min_nights": 1, "price": ""}]
        )

    collect(
        {"austin": ["1"]},
        tmp_path,
        now=NOW,
        delay=0,
        jitter=0,
        fetch=fetch,
        sleep=lambda _seconds: None,
        max_listings=1,
    )
    collect(
        {"austin": ["1", "2"]},
        tmp_path,
        now=NOW,
        delay=0,
        jitter=0,
        fetch=fetch,
        sleep=lambda _seconds: None,
    )
    assert calls == ["1", "1", "2"]
    saved = load_snapshot_files(tmp_path.rglob("*.csv.gz"))
    assert set(saved["listing_id"]) == {"1", "2"}


def test_retries_back_off_and_a_dead_endpoint_falls_back(tmp_path: Path):
    sleeps: list[float] = []

    def fetch(_listing_id: str, _when: datetime) -> Decision:
        return Decision("challenge", "html body")

    def download(url: str, dest: Path) -> None:
        assert url == AUSTIN
        frame = pd.DataFrame(
            {
                "listing_id": [1, 99],
                "date": ["2026-09-17", "2026-09-18"],
                "available": ["t", "f"],
                "price": ["$100.00", "$80"],
                "minimum_nights": [2, 3],
            }
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dest, index=False, compression="gzip")

    html = f'<a href="{AUSTIN}">calendar</a>'
    report = collect(
        {"austin": ["1", "2"]},
        tmp_path,
        now=NOW,
        delay=0,
        jitter=0,
        fetch=fetch,
        sleep=sleeps.append,
        fallback_html=lambda: html,
        download=download,
    )
    assert report["mode"] == "inside_airbnb"
    assert report["blocked"] is True
    assert report["cities"]["austin"]["succeeded"] == 1
    assert sleeps == []
    stored = next(tmp_path.rglob("inside-airbnb/austin/*.csv.gz"))
    frame = pd.read_csv(stored)
    assert list(frame["listing_id"].astype(str)) == ["1"]
    assert int(frame.iloc[0]["available"]) == 1
    assert str(frame.iloc[0]["price"]) == "100"
    assert "one published calendar date" in " ".join(report["cities"]["austin"]["notes"])


def test_transport_retries_use_backoff_then_keep_going(tmp_path: Path):
    sleeps: list[float] = []
    calls = {"n": 0}

    def fetch(listing_id: str, _when: datetime) -> Decision:
        calls["n"] += 1
        if calls["n"] < 3:
            return Decision("retry", "http 503")
        return Decision(
            "ok", "ok", [{"date": "2026-11-20", "available": 0, "min_nights": 1, "price": "90"}]
        )

    report = collect(
        {"austin": ["7"]},
        tmp_path,
        now=NOW,
        delay=0,
        jitter=0,
        fetch=fetch,
        sleep=sleeps.append,
        max_retries=3,
    )
    assert sleeps == [5.0, 15.0]
    assert report["cities"]["austin"]["succeeded"] == 1
    assert report["blocked"] is False


def test_build_script_reads_a_snapshot(tmp_path: Path):
    from scripts.build_occupancy_labels import main

    snap = tmp_path / "daily" / "austin" / "2026-04-12.csv.gz"
    write_snapshot(
        snap,
        [
            {
                "listing_id": "5",
                "date": "2026-04-12",
                "available": 1,
                "min_nights": 1,
                "price": "",
                "fetched_at": "2026-04-12T12:00:00Z",
            }
        ],
    )
    out = tmp_path / "labels.csv"
    assert main(["--snapshots", str(tmp_path), "--out", str(out)]) == 0
    saved = pd.read_csv(out)
    assert saved.iloc[0]["listing_id"] == 5 or str(saved.iloc[0]["listing_id"]) == "5"
    assert saved.iloc[0]["city"] == "austin"
    assert int(saved.iloc[0]["unbooked_nights"]) == 1
