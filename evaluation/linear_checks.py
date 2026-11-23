"""Checks that describe the shipped linear model, and nothing else.

Confirmation seeds 19, 29, and 37 refit that same linear specification on a
new spatial split. They are not used to change a feature or a model.

The later-snapshot check scores the saved app model on a newer Inside Airbnb
file. A row is kept when its listing id and its host id are both absent from
the primary training rows. The local median is the one fit on those training
rows.

    python -m evaluation.linear_checks

The later files are read from /tmp/capscout-later and are not committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from data.cities import CITIES, STUDY_CITIES
from data.clean import RAW_COLUMNS, apply_filters
from features.frame import attach_base_features
from models.artifacts import load_artifact

from evaluation.metrics import regression_metrics
from evaluation.recheck import shipped_columns
from evaluation.study_data import collapse_building, prepare_city, repair_superhost
from evaluation.study_format import money, pct, r2
from evaluation.study_models import AreaMedian, make_linear
from evaluation.study_protocol import CONFIRM_SEEDS, PRIMARY_SEED
from evaluation.study_split import audit_split, split_frame

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LATER_DIR = Path("/tmp/capscout-later")
LATER_DATES = {
    "austin": "2026-09-17",
    "san-diego": "2026-09-20",
    "seattle": "2026-09-20",
}


def _id_key(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    return text


def _feature_frame(frame: pd.DataFrame, city: str) -> pd.DataFrame:
    out = attach_base_features(frame, city)
    out = repair_superhost(out)
    out = collapse_building(out)
    return out


def _download_later() -> None:
    import urllib.request

    LATER_DIR.mkdir(parents=True, exist_ok=True)
    for city, date in LATER_DATES.items():
        dest = LATER_DIR / f"{city}.csv.gz"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        meta = CITIES[city]
        url = (
            "https://data.insideairbnb.com/united-states/"
            f"{meta['ia_state']}/{meta['ia_city']}/{date}/data/listings.csv.gz"
        )
        print(f"Downloading {city} {date}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "CapScout research"})
        with urllib.request.urlopen(request, timeout=180) as response:
            dest.write_bytes(response.read())


def _load_later(city: str) -> pd.DataFrame:
    path = LATER_DIR / f"{city}.csv.gz"
    header = pd.read_csv(path, nrows=0).columns
    usecols = [column for column in RAW_COLUMNS if column in header]
    raw = pd.read_csv(path, usecols=usecols, dtype={"id": str, "host_id": str}, low_memory=False)
    frame, _report = apply_filters(raw, city)
    return _feature_frame(frame, city)


def _score_block(y, pred, baseline) -> dict:
    model = regression_metrics(y, pred)
    base = regression_metrics(y, baseline)
    removed = (base["mae"] - model["mae"]) / base["mae"] if base["mae"] > 0 else float("nan")
    return {
        "n": int(model["n"]),
        "r2": model["r2"],
        "mae": model["mae"],
        "median_ape": model["median_ape"],
        "baseline_mae": base["mae"],
        "error_removed": float(removed),
    }


def confirmation(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    rows = []
    by_city = {}
    for city in STUDY_CITIES:
        frame = frames[city]
        columns = shipped_columns(city)
        y = frame["price_clean"].to_numpy(dtype=float)
        city_rows = []
        for seed in CONFIRM_SEEDS:
            split = split_frame(frame, city, int(seed))
            audit = audit_split(frame, split["train_idx"], split["test_idx"])
            if not audit["passed"]:
                raise RuntimeError(f"{city} seed {seed} failed the audit: {audit}")
            train_idx = split["train_idx"]
            test_idx = split["test_idx"]
            model = make_linear(columns, kind="log", high=None)
            model.fit(frame.iloc[train_idx], y[train_idx])
            pred = model.predict(frame.iloc[test_idx])
            metrics = regression_metrics(y[test_idx], pred)
            city_rows.append(metrics)
            for index, row_index in enumerate(test_idx):
                rows.append(
                    {
                        "city": city,
                        "seed": int(seed),
                        "y_price": float(y[row_index]),
                        "pred": float(pred[index]),
                    }
                )
            print(
                f"confirm {city} seed {seed} n {metrics['n']} "
                f"R2 {metrics['r2']:.3f} MAE {metrics['mae']:.2f}",
                flush=True,
            )
        by_city[city] = {
            "seeds": [int(seed) for seed in CONFIRM_SEEDS],
            "r2": [row["r2"] for row in city_rows],
            "mae": [row["mae"] for row in city_rows],
            "r2_spread": float(
                max(row["r2"] for row in city_rows) - min(row["r2"] for row in city_rows)
            ),
            "mae_spread": float(
                max(row["mae"] for row in city_rows) - min(row["mae"] for row in city_rows)
            ),
        }
    return pd.DataFrame(rows), by_city


def later_snapshot(frames: dict[str, pd.DataFrame], splits: dict) -> tuple[pd.DataFrame, dict]:
    _download_later()
    rows = []
    by_city = {}
    for city in STUDY_CITIES:
        study = frames[city]
        train_idx = splits[city]["train_idx"]
        train = study.iloc[train_idx]
        train_ids = set(_id_key(train["id"]))
        train_hosts = set(train["host_id"].astype(str))
        later = _load_later(city)
        ids = _id_key(later["id"])
        hosts = later["host_id"].astype(str)
        keep = ~ids.isin(train_ids) & ~hosts.isin(train_hosts)
        scored = later.loc[keep].reset_index(drop=True)
        model = load_artifact(city)["model"]
        area = AreaMedian().fit(
            train["neighbourhood_cleansed"],
            train["room_type"],
            train["price_clean"].to_numpy(dtype=float),
        )
        pred = model.predict(scored)
        baseline = area.predict(scored["neighbourhood_cleansed"], scored["room_type"])
        y = scored["price_clean"].to_numpy(dtype=float)
        block = _score_block(y, pred, baseline)
        block["snapshot"] = LATER_DATES[city]
        by_city[city] = block
        for price, guess, base in zip(y, pred, baseline, strict=True):
            rows.append(
                {
                    "city": city,
                    "y_price": float(price),
                    "pred": float(guess),
                    "baseline": float(base),
                }
            )
        print(
            f"later {city} n {block['n']} R2 {block['r2']:.3f} "
            f"MAE {block['mae']:.2f} removed {block['error_removed']:.1%}",
            flush=True,
        )
    pooled = pd.DataFrame(rows)
    by_city["every_home"] = _score_block(
        pooled["y_price"].to_numpy(),
        pooled["pred"].to_numpy(),
        pooled["baseline"].to_numpy(),
    )
    return pooled, by_city


def _table_confirmation(primary: dict, checks: dict) -> str:
    lines = [
        "| City | Primary R² | Other seeds R² | Primary MAE | Other seeds MAE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    labels = {"austin": "Austin, TX", "san-diego": "San Diego, CA", "seattle": "Seattle, WA"}
    for city in STUDY_CITIES:
        block = checks[city]
        r2s = ", ".join(r2(value) for value in block["r2"])
        maes = ", ".join(money(value) for value in block["mae"])
        lines.append(
            "| "
            + " | ".join(
                [
                    labels[city],
                    r2(primary[city]["r2"]),
                    f"{r2s} (spread {r2(block['r2_spread'])})",
                    money(primary[city]["mae"]),
                    f"{maes} (spread {money(block['mae_spread'])})",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _table_later(checks: dict) -> str:
    lines = [
        "| City | Snapshot | Homes | R² | Average miss | Median error | Local median miss | Less error |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {"austin": "Austin", "san-diego": "San Diego", "seattle": "Seattle"}
    for city in STUDY_CITIES:
        block = checks[city]
        lines.append(
            "| "
            + " | ".join(
                [
                    labels[city],
                    block["snapshot"],
                    f"{block['n']:,}",
                    r2(block["r2"]),
                    money(block["mae"]),
                    pct(block["median_ape"]),
                    money(block["baseline_mae"]),
                    pct(100.0 * block["error_removed"]),
                ]
            )
            + " |"
        )
    every = checks["every_home"]
    lines.append(
        "| Every home once | | "
        + " | ".join(
            [
                f"{every['n']:,}",
                r2(every["r2"]),
                money(every["mae"]),
                pct(every["median_ape"]),
                money(every["baseline_mae"]),
                pct(100.0 * every["error_removed"]),
            ]
        )
        + " |"
    )
    return "\n".join(lines)


def main() -> None:
    print(f"Loading the study cities. Primary seed {PRIMARY_SEED}.", flush=True)
    frames = {}
    splits = {}
    for city in STUDY_CITIES:
        frame, _report = prepare_city(city)
        frames[city] = frame
        splits[city] = split_frame(frame, city, PRIMARY_SEED)
    adopted = json.loads((REPORTS / "study_adopted.json").read_text())
    primary = {
        city: {"r2": adopted["cities"][city]["r2"], "mae": adopted["cities"][city]["mae"]}
        for city in STUDY_CITIES
    }
    print("Confirmation seeds", flush=True)
    confirm_rows, confirm = confirmation(frames)
    print("Later snapshots", flush=True)
    later_rows, later = later_snapshot(frames, splits)
    confirm_path = REPORTS / "linear_confirmation.csv"
    later_path = REPORTS / "linear_later.csv"
    confirm_rows.to_csv(confirm_path, index=False)
    later_rows.to_csv(later_path, index=False)
    payload = {
        "model": "linear",
        "primary_seed": PRIMARY_SEED,
        "confirm_seeds": [int(seed) for seed in CONFIRM_SEEDS],
        "confirmation": confirm,
        "later": later,
        "confirmation_table": _table_confirmation(primary, confirm),
        "later_table": _table_later(later),
    }
    (REPORTS / "linear_checks.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(payload["confirmation_table"])
    print(payload["later_table"])


if __name__ == "__main__":
    main()
