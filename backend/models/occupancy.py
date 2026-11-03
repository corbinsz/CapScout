"""Occupancy targets that do not treat a blocked night as a booking.

Inside Airbnb's San Francisco model, documented at
https://insideairbnb.com/data-assumptions/ :

- Half of stays are assumed to leave a review (a review rate of 50%).
- Estimated bookings = reviews in the last 12 months / 0.5.
- Nights booked = bookings x length of stay.
- Length of stay is the city's average, or the listing's minimum nights when
  that minimum is longer. A minimum above 14 nights is capped, because a
  30-night rule is a monthly rental and would otherwise invent a full hotel
  calendar from a couple of reviews.
- The rate is nights / 365, and it is not allowed to exceed 70%.

The availability calendar is used only as a check. A listing with no recent
reviews and a full near-term calendar is treated as blocked, not booked.
"""
REVIEW_RATE = 0.50
OCCUPANCY_CAP = 0.70
MAX_STAY_NIGHTS = 14.0


def length_of_stay(minimum_nights, city_avg_stay):
    """Nights per booking. A long minimum raises the stay, then stops at 14."""
    stay = float(city_avg_stay) if city_avg_stay and city_avg_stay > 0 else 3.0
    if minimum_nights is None:
        return stay
    try:
        minimum = float(minimum_nights)
    except (TypeError, ValueError):
        return stay
    if minimum != minimum or minimum <= 0:
        return stay
    if minimum > stay:
        stay = minimum
    return min(stay, MAX_STAY_NIGHTS)


def san_francisco_occupancy(reviews_ltm, minimum_nights, city_avg_stay):
    """Booked-night share of the year from reviews. Zero reviews means zero."""
    try:
        reviews = float(reviews_ltm or 0)
    except (TypeError, ValueError):
        reviews = 0.0
    if reviews != reviews or reviews < 0:
        reviews = 0.0
    stay = length_of_stay(minimum_nights, city_avg_stay)
    nights = (reviews / REVIEW_RATE) * stay
    rate = nights / 365.0
    if rate < 0:
        return 0.0
    if rate > OCCUPANCY_CAP:
        return OCCUPANCY_CAP
    return rate


def calendar_occupancy(availability_30):
    """Share of the next 30 nights the calendar marks unavailable.

    Unavailable can mean booked or blocked. This is the old target.
    """
    try:
        available = float(availability_30)
    except (TypeError, ValueError):
        return None
    if available != available:
        return None
    rate = 1.0 - (available / 30.0)
    return min(1.0, max(0.0, rate))


def blocked_not_booked(availability_30, reviews_ltm):
    """True when the next month looks full and nobody reviewed a stay."""
    try:
        available = float(availability_30)
        reviews = float(reviews_ltm or 0)
    except (TypeError, ValueError):
        return False
    if available != available or reviews != reviews:
        return False
    return available <= 0 and reviews <= 0
