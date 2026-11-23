"""Letter grades are ranks of predicted revenue, with the boundary taken upward."""

from models.grade import annual_revenue, grade_revenue, gross_yield

CUTS = {"p20": 1000.0, "p40": 2000.0, "p60": 3000.0, "p80": 4000.0}


def test_boundary_takes_the_higher_grade():
    assert grade_revenue(4000.0, CUTS)["letter"] == "A"
    assert grade_revenue(3999.99, CUTS)["letter"] == "B"
    assert grade_revenue(3000.0, CUTS)["letter"] == "B"
    assert grade_revenue(2000.0, CUTS)["letter"] == "C"
    assert grade_revenue(1000.0, CUTS)["letter"] == "D"
    assert grade_revenue(999.0, CUTS)["letter"] == "F"
    assert grade_revenue(1000.0, CUTS)["words"] == "Below average"


def test_revenue_is_rate_times_nights_and_yield_needs_a_price():
    assert annual_revenue(100.0, 50.0) == 5000.0
    assert gross_yield(5000.0, None) is None
    assert gross_yield(5000.0, 0) is None
    assert gross_yield(5000.0, 100000.0) == 0.05
