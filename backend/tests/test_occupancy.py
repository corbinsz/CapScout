from models.occupancy import (
    blocked_not_booked,
    calendar_occupancy,
    length_of_stay,
    san_francisco_occupancy,
)


def test_reviews_convert_at_a_fifty_percent_rate():
    # 10 reviews -> 20 bookings. City stay is 3 nights. 60 / 365.
    rate = san_francisco_occupancy(10, minimum_nights=1, city_avg_stay=3)
    assert abs(rate - (20 * 3) / 365) < 1e-9


def test_minimum_nights_raise_the_stay_and_a_monthly_rule_is_capped():
    assert length_of_stay(5, 3) == 5
    assert length_of_stay(30, 3) == 14
    capped = san_francisco_occupancy(10, minimum_nights=30, city_avg_stay=3)
    assert abs(capped - min(0.70, (20 * 14) / 365)) < 1e-9


def test_occupancy_cannot_exceed_seventy_percent():
    assert san_francisco_occupancy(200, minimum_nights=2, city_avg_stay=3) == 0.70


def test_a_full_calendar_with_no_reviews_is_not_booked():
    assert calendar_occupancy(0) == 1
    assert san_francisco_occupancy(0, minimum_nights=2, city_avg_stay=3) == 0
    assert blocked_not_booked(0, 0) is True
    assert blocked_not_booked(0, 4) is False
