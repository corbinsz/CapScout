"""Open the local app in one command.

From the repository root, after the libraries are installed:

    python -m app.launch

The three Inside Airbnb snapshots are downloaded the first time they are
missing. Streamlit then listens on 127.0.0.1 port 8501. Nothing is published.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from data.cities import STUDY_CITIES
from data.ingest import raw_city_dir

ROOT = Path(__file__).resolve().parents[1]


def missing_snapshots() -> list[str]:
    """Cities whose study snapshot is not on disk yet."""
    missing = []
    for city in STUDY_CITIES:
        if not (raw_city_dir(city) / "listings.csv.gz").exists():
            missing.append(city)
    return missing


def main() -> None:
    missing = missing_snapshots()
    if missing:
        from data.download import main as download

        download(missing)
    app = ROOT / "app" / "streamlit_app.py"
    # Replace this process so the app is the one command the reader started.
    os_exec = os.execvp
    os_exec(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8501",
            "--browser.gatherUsageStats",
            "false",
        ],
    )


if __name__ == "__main__":
    main()
