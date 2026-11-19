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
listing has not failed, it has not been observed. A booked-night rate built
from repeated calendars lives in data/calendar_labels.py. It is not this proxy,
and it is not yet the training target.
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
# Inside Airbnb's written fallback when a city has no published average
# stay. The hand-set figures in data/cities.py are not that statement.
STATED_STAY_NIGHTS = 3.0
DAYS_IN_YEAR = 365.0
# 70% of a year. The same cap as the monthly rate, on an annual count.
ANNUAL_NIGHTS_CAP = OCCUPANCY_CAP * DAYS_IN_YEAR


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


def booked_nights_ltm(
    reviews_ltm: np.ndarray | pd.Series,
    minimum_nights: np.ndarray | pd.Series,
    average_stay_nights: float = STATED_STAY_NIGHTS,
) -> np.ndarray:
    """Estimated booked nights over the last 12 months.

    ``reviews_ltm`` is Inside Airbnb's ``number_of_reviews_ltm``. Bookings
    are reviews divided by the 50% review rate. Stay length is the stated
    average, or the listing's minimum nights when that minimum is longer
    and not above ``MAX_MIN_NIGHTS``. The count is capped at 70% of 365.
    Zero or missing reviews are unobserved, not zero nights.
    """
    reviews = np.asarray(reviews_ltm, dtype=float)
    minimum = np.asarray(minimum_nights, dtype=float)
    stay = np.full(reviews.shape, float(average_stay_nights), dtype=float)
    use_minimum = (
        np.isfinite(minimum) & (minimum > average_stay_nights) & (minimum <= MAX_MIN_NIGHTS)
    )
    stay = np.where(use_minimum, minimum, stay)
    nights = (reviews / REVIEW_RATE) * stay
    nights = np.minimum(nights, ANNUAL_NIGHTS_CAP)
    observed = np.isfinite(reviews) & (reviews > 0)
    return np.where(observed, nights, np.nan)
