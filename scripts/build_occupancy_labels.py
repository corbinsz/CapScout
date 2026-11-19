"""Turn accumulated calendar snapshots into observed booked-night rates.

The rules are in ``data/calendar_labels.py`` and ``docs/CALENDAR.md``.
The rate uses nights that have already occurred and that the rules can call
booked or still open. Blocked nights, censored nights, and future nights
are counted beside it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from data.calendar_labels import label_frame, load_snapshot_files
from data.calendar_sample import load_sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build observed booked-night rates from calendar snapshots."
    )
    parser.add_argument(
        "--snapshots", type=Path, required=True, help="Directory of collector csv.gz files."
    )
    parser.add_argument("--out", type=Path, required=True, help="Per-listing summary CSV.")
    parser.add_argument(
        "--night-detail", type=Path, default=None, help="Optional per-night label CSV."
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=None,
        help="Panel JSON. Default: the committed sample. Listings outside it are dropped.",
    )
    parser.add_argument(
        "--all-listings",
        action="store_true",
        help="Keep every listing in the snapshots, including ids that are not in the panel.",
    )
    args = parser.parse_args(argv)

    paths = sorted(
        path
        for path in args.snapshots.rglob("*.csv.gz")
        if not path.name.endswith(".partial.csv.gz")
    )
    frame = load_snapshot_files(paths)
    if not args.all_listings and not frame.empty:
        panel = load_sample(args.sample)
        wanted = {str(listing_id) for ids in panel.values() for listing_id in ids}
        frame = frame[frame["listing_id"].astype(str).isin(wanted)].copy()
    nights, summary = label_frame(frame)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = summary.copy()
    if "booked_rate" in written.columns:
        written["booked_rate"] = written["booked_rate"].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
        )
    written.to_csv(args.out, index=False)
    if args.night_detail is not None:
        args.night_detail.parent.mkdir(parents=True, exist_ok=True)
        nights.to_csv(args.night_detail, index=False)
    print(f"listings {len(summary)} nights {len(nights)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
