"""Inside Airbnb's San Francisco occupancy model, as a proxy — not a label.

The calendar file cannot tell a booked night from a night the host blocked,
so this study does not treat availability as occupancy. Inside Airbnb's
published alternative (the "San Francisco model", documented at
https://insideairbnb.com/data-assumptions/) estimates bookings from reviews:

* 50% of stays are assumed to leave a review (their compromise between a
  72% figure attributed to Airbnb and a 30.5% figure from a New York
  comparison).
* Nights per booking are the city's average length of stay, or the listing's
  minimum nights when that minimum is longer.
* Occupancy is estimated nights divided by 30, capped at 70% so a listing
  with an unusually high review rate cannot imply a full hotel.

The output is a model of review activity, not a count of bookings. Listings
with no reviews are left missing rather than coded as 0% occupied: a new
listing has not failed, it has not been observed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REVIEW_RATE = 0.50
OCCUPANCY_CAP = 0.70
# A minimum-stay field above this is almost always a data error or a
# monthly lease mis-filed as a nightly listing. Past this point the
# documented rule (use minimum nights as the stay length) would pin every
# such row to the 70% cap. Fall back to the city average instead.
MAX_MIN_NIGHTS = 90


def san_francisco_occupancy(
    reviews_per_month: np.ndarray | pd.Series,
    minimum_nights: np.ndarray | pd.Series,
    average_stay_nights: float,
) -> np.ndarray:
    """Estimated occupancy in [0, 0.70], NaN where reviews are missing or zero."""
    reviews = np.asarray(reviews_per_month, dtype=float)
    minimum = np.asarray(minimum_nights, dtype=float)
    stay = np.full(reviews.shape, float(average_stay_nights), dtype=float)
    use_minimum = (
        np.isfinite(minimum) & (minimum > average_stay_nights) & (minimum <= MAX_MIN_NIGHTS)
    )
    stay = np.where(use_minimum, minimum, stay)

    bookings = reviews / REVIEW_RATE
    occupancy = bookings * stay / 30.0
    occupancy = np.minimum(occupancy, OCCUPANCY_CAP)

    observed = np.isfinite(reviews) & (reviews > 0)
    occupancy = np.where(observed, occupancy, np.nan)
    return occupancy
