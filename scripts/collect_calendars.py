"""Collect one daily public availability snapshot for the fixed city panel.

The panel is ``data/calendar_sample.json``. Snapshots land in ``--out-dir``
and are meant for the ``calendar-data`` branch, not for main. See
``docs/CALENDAR.md``.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.calendar_collect import main

if __name__ == "__main__":
    raise SystemExit(main())
