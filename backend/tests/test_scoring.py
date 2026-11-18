"""Checks for the CapScout score math in services/scoring_service.py."""

import pytest

from services.scoring_service import (
    calculate_annual_mortgage,
    calculate_operating_expenses,
    score_property,
)


def test_zero_interest_mortgage_is_principal_over_term():
    assert calculate_annual_mortgage(120_000, rate=0, years=30) == pytest.approx(4_000)


def test_mortgage_payment_matches_standard_formula():
    loan = 200_000
    annual_rate = 0.07
    years = 30
    monthly_rate = annual_rate / 12
    n_payments = years * 12
    monthly = (
        loan
        * (monthly_rate * (1 + monthly_rate) ** n_payments)
        / ((1 + monthly_rate) ** n_payments - 1)
    )
    assert calculate_annual_mortgage(loan, annual_rate, years) == pytest.approx(monthly * 12)
    assert calculate_annual_mortgage(loan, 0.08, years) > calculate_annual_mortgage(
        loan, annual_rate, years
    )


def test_operating_expenses_itemizes_austin_assumptions():
    city = {
        "cleaning_per_turnover": 125,
        "avg_stay_nights": 3.2,
        "base_utilities_monthly": 275,
    }
    revenue = 50_000
    occupancy = 0.5
    price = 300_000
    turnovers = (occupancy * 365) / 3.2
    expected = (
        revenue * 0.03
        + turnovers * 125
        + turnovers * 25
        + 275 * 12 * (0.6 + 0.4 * occupancy)
        + (75 + 50) * 12
        + price * 0.01
    )
    total, breakdown = calculate_operating_expenses(revenue, occupancy, price, city)
    assert total == pytest.approx(round(expected, 2))
    assert breakdown["turnovers"] == pytest.approx(round(turnovers, 1))
    assert breakdown["platform_fees"] == pytest.approx(1_500)


def test_score_uses_city_tax_and_permit_and_bounds_the_total():
    missing = score_property(0, 80_000, 0.6, city_key="austin")
    assert missing["score"] == 0

    # 20% gross yield and 75% occupancy sit on the top of those two curves.
    result = score_property(400_000, 80_000, 0.75, city_key="hawaii")
    assert result["financials"]["down_payment"] == pytest.approx(80_000)
    assert result["financials"]["tax_rate"] == pytest.approx(0.3)
    assert result["financials"]["str_permit_annual"] == pytest.approx(500)
    assert result["financials"]["revenue_to_price"] == pytest.approx(20.0)
    assert result["breakdown"]["yield_score"] == pytest.approx(25.0)
    assert result["breakdown"]["occupancy_score"] == pytest.approx(10.0)

    parts = result["breakdown"]
    assert result["score"] == pytest.approx(
        min(100, sum(parts.values())),
        abs=0.15,
    )
    assert 0 <= result["score"] <= 100

    richer = score_property(400_000, 120_000, 0.75, city_key="hawaii")
    assert richer["score"] >= result["score"]


def test_negative_cash_flow_does_not_leave_the_zero_to_hundred_range():
    result = score_property(900_000, 5_000, 0.2, city_key="austin")
    assert 0 <= result["score"] <= 100
    assert result["financials"]["annual_cashflow"] < 0
    assert result["breakdown"]["roi_score"] < 5
    assert result["financials"]["tax_rate"] == pytest.approx(1.8)
