"""Daily public-calendar collection, and the Inside Airbnb fallback.

One listing, one request, then a pause. The client identifies itself, retries
ordinary server errors with backoff, and stops the run when the response is a
block or a challenge page. It does not log in, solve a captcha, rotate
proxies, or impersonate a browser.

The calendar this calls is the public availability query the listing page
uses. The persisted-query hash and the web client key are the ones that
query has used in public; they are not credentials. When that query stops
returning a calendar, or the first request is challenged, the run switches
to Inside Airbnb's published calendar.csv.gz files and does not keep asking.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from data.calendar_sample import SAMPLE_CITIES, load_sample
from data.calendar_sample import listing_id as normalize_listing_id
from data.cities import CITIES

USER_AGENT = (
    "CapScoutCalendarCollector/1.0 "
    "(personal research; public calendar snapshot; github.com/corbinsz/CapScout)"
)
INSIDE_AIRBNB_USER_AGENT = "CapScout personal research (CC BY 4.0 Inside Airbnb snapshot)"

# Public web client key and persisted query, as sent by the listing page.
# Not a secret. A mismatch means the public query changed; the run falls back.
PUBLIC_API_KEY = "d306zoyjsyarp7ifhu67rjxn52tv0t20"
CALENDAR_HASH = "b23335819df0dc391a338d665e2ee2f5d3bff19181d05c0b39bc6c5aac403914"
CALENDAR_ENDPOINT = "https://www.airbnb.com/api/v3/PdpAvailabilityCalendar/" + CALENDAR_HASH
INSIDE_AIRBNB_INDEX = "https://insideairbnb.com/get-the-data/"

COLUMNS = ["listing_id", "date", "available", "min_nights", "price", "fetched_at"]
BACKOFF_SECONDS = (5.0, 15.0, 45.0)
DEFAULT_DELAY = 2.0
DEFAULT_JITTER = 1.0
MAX_RETRIES = 3

CHALLENGE_MARKERS = (
    "captcha",
    "datadome",
    "cf-browser-verification",
    "unusual traffic",
    "verify you are human",
    "px-captcha",
    "challenge-platform",
    "just a moment",
    "cdn-cgi/challenge",
)
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
CALENDAR_LINK = re.compile(
    r"https://data\.insideairbnb\.com/united-states/"
    r"(?P<state>[a-z-]+)/(?P<city>[a-z0-9-]+)/(?P<date>\d{4}-\d{2}-\d{2})/data/calendar\.csv\.gz"
)

Sleep = Callable[[float], None]
Fetch = Callable[[str, datetime], "Decision"]


@dataclass
class Decision:
    kind: str
    reason: str
    days: list[dict] | None = None


@dataclass
class CityResult:
    city: str
    mode: str
    attempted: int = 0
    succeeded: int = 0
    missing: int = 0
    failed: int = 0
    blocked: bool = False
    block_reason: str = ""
    complete: bool = False
    rows: int = 0
    path: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "city": self.city,
            "mode": self.mode,
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "missing": self.missing,
            "failed": self.failed,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "complete": self.complete,
            "rows": self.rows,
            "path": self.path,
            "notes": self.notes,
        }


def calendar_url(listing_id: str, month: int, year: int) -> str:
    variables = json.dumps(
        {
            "request": {
                "count": 12,
                "listingId": str(listing_id),
                "month": int(month),
                "year": int(year),
                "returnPropertyLevelCalendarIfApplicable": False,
            }
        },
        separators=(",", ":"),
    )
    extensions = json.dumps(
        {"persistedQuery": {"version": 1, "sha256Hash": CALENDAR_HASH}},
        separators=(",", ":"),
    )
    query = urllib.parse.urlencode(
        {
            "operationName": "PdpAvailabilityCalendar",
            "locale": "en",
            "currency": "USD",
            "variables": variables,
            "extensions": extensions,
        }
    )
    return f"{CALENDAR_ENDPOINT}?{query}"


def parse_price(value: object) -> str:
    """A plain number, or blank when the calendar did not include a price."""
    if value is None:
        return ""
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null", "nan"}:
        return ""
    cleaned = re.sub(r"[^0-9.]", "", text)
    if cleaned in {"", "."}:
        return ""
    try:
        number = float(cleaned)
    except ValueError:
        return ""
    if number <= 0:
        return ""
    if number.is_integer():
        return str(int(number))
    rendered = f"{number:.2f}".rstrip("0").rstrip(".")
    return rendered


def parse_calendar_payload(payload: dict) -> list[dict] | None:
    """Days from a public calendar payload. None when the payload has no calendar."""
    try:
        months = payload["data"]["merlin"]["pdpAvailabilityCalendar"]["calendarMonths"]
    except (KeyError, TypeError):
        return None
    if not months:
        return []
    days: list[dict] = []
    for month in months:
        for day in month.get("days") or []:
            if not day or not day.get("calendarDate"):
                continue
            if day.get("available") is None:
                continue
            min_nights = day.get("minNights")
            price = day.get("price") or {}
            days.append(
                {
                    "date": day["calendarDate"],
                    "available": 1 if day["available"] else 0,
                    "min_nights": "" if min_nights is None else int(min_nights),
                    "price": parse_price(price.get("localPriceFormatted")),
                }
            )
    return days


def _persisted_query_missing(payload: dict) -> bool:
    for error in payload.get("errors") or []:
        message = str(error.get("message", "")).lower()
        if "persistedquery" in message.replace(" ", "") or "persisted query" in message:
            return True
    return False


def decide_response(status: int | None, content_type: str, body: bytes) -> Decision:
    """Classify one HTTP response. Challenge and unusable both stop the live run."""
    if status is None:
        return Decision("retry", "no response")
    if status in RETRY_STATUSES:
        return Decision("retry", f"http {status}")
    if status in {401, 403}:
        return Decision("challenge", f"http {status}")
    snippet = body[:8000].decode("utf-8", errors="replace")
    lowered = snippet.lower()
    kind = (content_type or "").lower()
    html = "html" in kind or lowered.lstrip().startswith(("<!doctype html", "<html"))
    if html:
        return Decision("challenge", "html body")
    if any(marker in lowered for marker in CHALLENGE_MARKERS):
        return Decision("challenge", "challenge page")
    if status == 404:
        return Decision("missing", "http 404")
    if status == 400:
        return Decision("unusable", "http 400")
    if status != 200:
        return Decision("unusable", f"http {status}")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return Decision("challenge", "non-json body")
    if _persisted_query_missing(payload):
        return Decision("unusable", "persisted query not found")
    days = parse_calendar_payload(payload)
    if not days:
        return Decision("missing", "no calendar days")
    return Decision("ok", "ok", days)


def fetch_public_calendar(
    session: requests.Session, listing_id: str, when: datetime, timeout: float = 30
) -> Decision:
    url = calendar_url(listing_id, when.month, when.year)
    try:
        response = session.get(url, timeout=timeout)
    except requests.RequestException as exc:
        return Decision("retry", type(exc).__name__)
    return decide_response(
        response.status_code, response.headers.get("content-type", ""), response.content
    )


def public_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "X-Airbnb-Api-Key": PUBLIC_API_KEY,
        }
    )
    return session


def inside_airbnb_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": INSIDE_AIRBNB_USER_AGENT, "Accept": "*/*"})
    return session


def _pause(delay: float, jitter: float, rng: random.Random, sleep: Sleep) -> None:
    wait = max(0.0, delay) + (rng.random() * max(0.0, jitter))
    if wait:
        sleep(wait)


def _log(message: str) -> None:
    print(message, flush=True)


def snapshot_rows(listing_id: str, days: list[dict], fetched_at: str) -> list[dict]:
    rows = []
    for day in days:
        rows.append(
            {
                "listing_id": str(listing_id),
                "date": day["date"],
                "available": int(day["available"]),
                "min_nights": day.get("min_nights", ""),
                "price": day.get("price", ""),
                "fetched_at": fetched_at,
            }
        )
    return rows


def _append_partial(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def _read_partial(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_snapshot(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (str(row["listing_id"]), str(row["date"])))
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)


def succeeded_ids(rows: list[dict]) -> set[str]:
    return {str(row["listing_id"]) for row in rows}


def _stamp(when: datetime) -> str:
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def collect_public_city(
    city: str,
    listing_ids: Sequence[str],
    out_dir: Path,
    *,
    now: datetime,
    delay: float,
    jitter: float,
    rng: random.Random,
    fetch: Fetch,
    sleep: Sleep,
    max_retries: int = MAX_RETRIES,
    max_listings: int | None = None,
) -> tuple[CityResult, list[dict], Decision | None]:
    """Fetch one city. A challenge or a dead endpoint stops the city immediately."""
    day = now.astimezone(UTC).date().isoformat()
    final_path = out_dir / "daily" / city / f"{day}.csv.gz"
    partial_path = out_dir / "daily" / city / f"{day}.partial.csv"
    meta_path = out_dir / "daily" / city / f"{day}.json"
    result = CityResult(city=city, mode="airbnb_public", attempted=len(listing_ids))

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stored_cap = meta.get("max_listings")
        # A missing cap is the full panel. A smaller manual cap must not replace it.
        full_on_disk = stored_cap is None
        capped_covers_request = (
            stored_cap is not None and max_listings is not None and int(stored_cap) >= max_listings
        )
        if meta.get("complete") and final_path.exists() and (full_on_disk or capped_covers_request):
            result.complete = True
            result.succeeded = int(meta.get("succeeded", 0))
            result.missing = int(meta.get("missing", 0))
            result.failed = int(meta.get("failed", 0))
            result.rows = int(meta.get("rows", 0))
            result.path = str(final_path)
            result.notes.append("already collected today")
            return result, [], None

    rows = _read_partial(partial_path)
    done = succeeded_ids(rows)
    stop: Decision | None = None
    missing: list[str] = []
    failed: list[str] = []
    first_request = True
    transport_failures = 0
    for listing_id in listing_ids:
        if listing_id in done:
            continue
        if not first_request:
            _pause(delay, jitter, rng, sleep)
        first_request = False
        decision = Decision("retry", "not attempted")
        for attempt in range(max_retries + 1):
            decision = fetch(listing_id, now)
            if decision.kind == "retry" and attempt < max_retries:
                sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
                continue
            break
        fetched_at = _stamp(datetime.now(UTC))
        if decision.kind == "ok" and decision.days is not None:
            transport_failures = 0
            new_rows = snapshot_rows(listing_id, decision.days, fetched_at)
            _append_partial(partial_path, new_rows)
            rows.extend(new_rows)
            done.add(listing_id)
            _log(
                f"{city} {len(done)}/{len(listing_ids)} ok {listing_id} {len(decision.days)} nights"
            )
            continue
        if decision.kind == "missing":
            transport_failures = 0
            missing.append(listing_id)
            _log(f"{city} missing {listing_id}: {decision.reason}")
            continue
        if decision.kind in {"challenge", "unusable"}:
            stop = decision
            _log(f"{city} stopped on {decision.kind}: {decision.reason}")
            break
        if decision.kind == "retry":
            transport_failures += 1
            failed.append(listing_id)
            _log(f"{city} transport failure {listing_id}: {decision.reason}")
            # Three listings in a row never answered. Stop rather than hammer a dead route.
            if transport_failures >= 3:
                stop = decision
                _log(f"{city} stopped after {transport_failures} unanswered listings")
                break
            continue
        failed.append(listing_id)
        _log(f"{city} failed {listing_id}: {decision.reason}")

    result.succeeded = len(done)
    result.missing = len(missing)
    result.failed = len(failed)
    result.rows = len(rows)
    result.blocked = stop is not None
    result.block_reason = "" if stop is None else f"{stop.kind}: {stop.reason}"
    result.complete = stop is None
    if stop is not None:
        result.notes.append("stopped before the rest of the panel")
    if rows:
        write_snapshot(final_path, rows)
        result.path = str(final_path)
    meta = result.as_dict()
    meta["date"] = day
    meta["max_listings"] = max_listings
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if result.complete and partial_path.exists():
        partial_path.unlink()
    return result, rows, stop


def calendar_links_by_city(
    html: str, cities: Sequence[str] = SAMPLE_CITIES
) -> dict[str, list[tuple[str, str]]]:
    """Dates and calendar.csv.gz URLs published on Inside Airbnb's index page."""
    wanted = {
        (CITIES[city]["ia_state"], CITIES[city]["ia_city"]): city
        for city in cities
        if city in CITIES
    }
    found: dict[str, dict[str, str]] = {city: {} for city in cities}
    for match in CALENDAR_LINK.finditer(html):
        key = (match.group("state"), match.group("city"))
        city = wanted.get(key)
        if city is None:
            continue
        found[city][match.group("date")] = match.group(0)
    return {city: sorted(dates.items()) for city, dates in found.items()}


def _optional_int(value: object) -> int | str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return ""
    return int(float(text))


def inside_rows_for_sample(
    frame: pd.DataFrame, listing_ids: set[str], fetched_at: str
) -> list[dict]:
    """Filter an Inside Airbnb calendar table down to the panel and our columns."""
    if frame.empty:
        return []
    working = frame.copy()
    working.columns = [str(column).strip().lower() for column in working.columns]
    required = {"listing_id", "date", "available"}
    if not required.issubset(working.columns):
        missing = ", ".join(sorted(required - set(working.columns)))
        raise ValueError(f"Inside Airbnb calendar is missing {missing}")
    working["listing_id"] = working["listing_id"].map(
        lambda value: normalize_listing_id(value) or ""
    )
    working = working[working["listing_id"].isin(listing_ids)]
    min_column = "minimum_nights" if "minimum_nights" in working.columns else "min_nights"
    rows = []
    for record in working.to_dict(orient="records"):
        rows.append(
            {
                "listing_id": record["listing_id"],
                "date": str(record["date"])[:10],
                "available": (
                    1
                    if str(record["available"]).strip().lower() in {"t", "true", "1", "yes"}
                    else 0
                ),
                "min_nights": _optional_int(record[min_column]) if min_column in record else "",
                "price": parse_price(record["price"]) if "price" in record else "",
                "fetched_at": fetched_at,
            }
        )
    return rows


def read_inside_airbnb_calendar(path: Path, listing_ids: set[str], fetched_at: str) -> list[dict]:
    kept: list[pd.DataFrame] = []
    usecols = ["listing_id", "date", "available", "price", "minimum_nights"]
    for chunk in pd.read_csv(
        path,
        usecols=lambda name: name in usecols,
        dtype={"listing_id": "string"},
        chunksize=200_000,
    ):
        if chunk.empty or "listing_id" not in chunk.columns:
            continue
        chunk = chunk.copy()
        chunk["listing_id"] = chunk["listing_id"].map(
            lambda value: normalize_listing_id(value) or ""
        )
        chunk = chunk[chunk["listing_id"].isin(listing_ids)]
        if not chunk.empty:
            kept.append(chunk)
    if not kept:
        return []
    return inside_rows_for_sample(pd.concat(kept, ignore_index=True), listing_ids, fetched_at)


def collect_inside_airbnb(
    listing_ids_by_city: Mapping[str, Sequence[str]],
    out_dir: Path,
    *,
    html: str,
    download: Callable[[str, Path], None],
    sleep: Sleep,
    delay: float,
    jitter: float,
    rng: random.Random,
) -> list[CityResult]:
    """Write one compact snapshot per published Inside Airbnb calendar date.

    A single published date cannot show a night flipping. The files are still
    stored, dated as that scrape at 12:00 UTC, so the next published file can
    be diffed later. This path does not pretend they are a daily fetch.
    """
    links = calendar_links_by_city(html, tuple(listing_ids_by_city))
    results = []
    for city, listing_ids in listing_ids_by_city.items():
        result = CityResult(city=city, mode="inside_airbnb", attempted=len(listing_ids))
        city_links = links.get(city, [])
        if not city_links:
            result.notes.append("no calendar.csv.gz link on the Inside Airbnb index")
            result.complete = False
            results.append(result)
            continue
        if len(city_links) < 2:
            result.notes.append(
                "one published calendar date; a single file cannot show a night flipping from available to unavailable"
            )
        ids = set(listing_ids)
        written = 0
        for index, (snap_date, url) in enumerate(city_links):
            if index:
                _pause(delay, jitter, rng, sleep)
            dest = out_dir / "inside-airbnb" / city / f"{snap_date}.csv.gz"
            if dest.exists() and dest.stat().st_size > 0:
                existing = pd.read_csv(dest, usecols=["listing_id"], dtype={"listing_id": "string"})
                present = {str(value) for value in existing["listing_id"].dropna()}
                result.succeeded = len(present)
                result.missing = len(ids - present)
                result.rows += int(len(existing))
                result.path = str(dest)
                result.notes.append(f"{snap_date} already stored")
                written += 1
                continue
            raw_path = out_dir / "inside-airbnb" / city / f"{snap_date}.download.csv.gz"
            try:
                download(url, raw_path)
                rows = read_inside_airbnb_calendar(raw_path, ids, f"{snap_date}T12:00:00Z")
            except (OSError, ValueError, pd.errors.ParserError, requests.RequestException) as exc:
                result.notes.append(f"{snap_date} failed: {type(exc).__name__}")
                result.failed += 1
                continue
            finally:
                if raw_path.exists():
                    raw_path.unlink()
            if not rows:
                result.notes.append(f"{snap_date} had none of the sampled listings")
                continue
            write_snapshot(dest, rows)
            present = succeeded_ids(rows)
            result.succeeded = len(present)
            result.missing = len(ids - present)
            result.rows += len(rows)
            result.path = str(dest)
            written += 1
            _log(f"{city} inside airbnb {snap_date}: {len(present)} listings, {len(rows)} rows")
        result.complete = written > 0
        results.append(result)
    return results


def _download(session: requests.Session, url: str, dest: Path, timeout: float = 180) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    handle.write(chunk)


def _fallback(
    listing_ids_by_city: Mapping[str, Sequence[str]],
    out_dir: Path,
    *,
    sleep: Sleep,
    delay: float,
    jitter: float,
    rng: random.Random,
    fallback_html: Callable[[], str] | None,
    download: Callable[[str, Path], None] | None,
) -> list[CityResult]:
    ia_session = inside_airbnb_session()
    try:
        if fallback_html is None:
            page = ia_session.get(INSIDE_AIRBNB_INDEX, timeout=60)
            page.raise_for_status()
            html = page.text
        else:
            html = fallback_html()
        downloader = download or (lambda url, dest: _download(ia_session, url, dest))
        return collect_inside_airbnb(
            listing_ids_by_city,
            out_dir,
            html=html,
            download=downloader,
            sleep=sleep,
            delay=delay,
            jitter=jitter,
            rng=rng,
        )
    except (requests.RequestException, OSError) as exc:
        _log(f"Inside Airbnb fallback failed: {type(exc).__name__}: {exc}")
        return [
            CityResult(
                city=city,
                mode="inside_airbnb",
                attempted=len(listing_ids),
                notes=[f"fallback failed: {type(exc).__name__}"],
            )
            for city, listing_ids in listing_ids_by_city.items()
        ]
    finally:
        ia_session.close()


def collect(
    listing_ids_by_city: Mapping[str, Sequence[str]],
    out_dir: Path,
    *,
    now: datetime | None = None,
    delay: float = DEFAULT_DELAY,
    jitter: float = DEFAULT_JITTER,
    seed: int = 17,
    fetch: Fetch | None = None,
    sleep: Sleep = time.sleep,
    session: requests.Session | None = None,
    fallback_html: Callable[[], str] | None = None,
    download: Callable[[str, Path], None] | None = None,
    live: bool = True,
    max_retries: int = MAX_RETRIES,
    max_listings: int | None = None,
) -> dict:
    """Collect the panel. Fall back only when the live endpoint produced nothing."""
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    started = datetime.now(UTC)
    city_results: list[CityResult] = []
    stop: Decision | None = None
    live_rows = 0
    blocked = False

    if live:
        own_session = session is None and fetch is None
        if fetch is None:
            session = session or public_session()
            assert session is not None
            bound = session

            def _live_fetch(listing_id: str, when: datetime) -> Decision:
                return fetch_public_calendar(bound, listing_id, when)

            fetch = _live_fetch
        try:
            started_city = False
            for city, listing_ids in listing_ids_by_city.items():
                if stop is not None:
                    skipped = CityResult(
                        city=city, mode="airbnb_public", attempted=len(listing_ids)
                    )
                    skipped.blocked = True
                    skipped.block_reason = f"not attempted after {stop.kind}"
                    skipped.notes.append("live run already stopped")
                    city_results.append(skipped)
                    continue
                if started_city:
                    _pause(delay, jitter, rng, sleep)
                started_city = True
                result, _rows, stop = collect_public_city(
                    city,
                    listing_ids,
                    out_dir,
                    now=now,
                    delay=delay,
                    jitter=jitter,
                    rng=rng,
                    fetch=fetch,
                    sleep=sleep,
                    max_retries=max_retries,
                    max_listings=max_listings,
                )
                city_results.append(result)
                live_rows += result.rows
        finally:
            if own_session and session is not None:
                session.close()
        blocked = stop is not None or any(result.blocked for result in city_results)

    mode = "airbnb_public"
    if not live or (live_rows == 0 and blocked):
        mode = "inside_airbnb"
        if live:
            _log("Public calendar endpoint unusable. Falling back to Inside Airbnb calendar files.")
        else:
            _log("Fallback only. Not calling the public calendar endpoint.")
        city_results = _fallback(
            listing_ids_by_city,
            out_dir,
            sleep=sleep,
            delay=delay,
            jitter=jitter,
            rng=rng,
            fallback_html=fallback_html,
            download=download,
        )
        if not live:
            blocked = False
            stop = None

    report = {
        "started_at": _stamp(started),
        "finished_at": _stamp(datetime.now(UTC)),
        "mode": mode,
        "blocked": blocked,
        "block_reason": "" if stop is None else f"{stop.kind}: {stop.reason}",
        "delay_seconds": delay,
        "jitter_seconds": jitter,
        "cities": {result.city: result.as_dict() for result in city_results},
    }
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / "runs" / f"{stamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    for city, body in report["cities"].items():
        _log(
            f"{city}: mode={body['mode']} succeeded={body['succeeded']}/{body['attempted']} "
            f"missing={body['missing']} failed={body['failed']} blocked={body['blocked']}"
        )
    return report


def limit_sample(
    listing_ids_by_city: Mapping[str, Sequence[str]], max_listings: int | None
) -> dict[str, list[str]]:
    if max_listings is None:
        return {city: list(ids) for city, ids in listing_ids_by_city.items()}
    return {city: list(ids)[: max(0, max_listings)] for city, ids in listing_ids_by_city.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect one daily calendar snapshot for the fixed city panel."
    )
    parser.add_argument(
        "--sample", type=Path, default=None, help="Sample JSON. Default: data/calendar_sample.json"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("calendars"))
    parser.add_argument("--cities", nargs="+", default=list(SAMPLE_CITIES))
    parser.add_argument(
        "--max-listings",
        type=int,
        default=None,
        help="Cap listings per city. Default: the full panel.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Seconds between listings, before jitter.",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=DEFAULT_JITTER,
        help="Extra random seconds, uniform from 0 to this.",
    )
    parser.add_argument(
        "--write-sample", action="store_true", help="Redraw data/calendar_sample.json and exit."
    )
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Skip the live endpoint and read Inside Airbnb.",
    )
    args = parser.parse_args(argv)

    if args.write_sample:
        from data.calendar_sample import write_sample

        document = write_sample(args.sample)
        for city, body in document["cities"].items():
            _log(
                f"{city}: sampled {body['sampled']} of {body['source_rows']} across {body['strata']} strata"
            )
        return 0

    sample = load_sample(args.sample)
    unknown = [city for city in args.cities if city not in sample]
    if unknown:
        raise SystemExit(f"No sample for: {', '.join(unknown)}")
    selected = limit_sample({city: sample[city] for city in args.cities}, args.max_listings)

    report = collect(
        selected,
        args.out_dir,
        delay=args.delay,
        jitter=args.jitter,
        live=not args.fallback_only,
        max_listings=args.max_listings,
    )

    wrote = any(body.get("rows", 0) or body.get("complete") for body in report["cities"].values())
    _log(f"REPORT {report['report_path']}")
    return 0 if wrote else 2
