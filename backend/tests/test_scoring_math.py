"""Known-answer checks for cap rate, cash-on-cash, and debt coverage.

The mortgage rate is zero in the worked example so the payment is loan / term
and the rest of the arithmetic can be checked by hand.
"""
import pytest

from services.scoring_service import calculate_annual_mortgage, score_property


def test_zero_interest_mortgage_is_loan_over_term():
    annual = calculate_annual_mortgage(160000, rate=0, years=30)
    assert annual == pytest.approx(160000 / 30)


def test_cap_rate_cash_on_cash_and_dscr_match_a_hand_worked_austin_deal():
    # $200,000 home, 20% down, 0% interest, 30 years, Austin tax and insurance.
    # Revenue $40,000. Occupancy 50%. City stay 3.2 nights, cleaning $125.
    result = score_property(
        200000,
        40000,
        0.5,
        city_key='austin',
        mortgage_rate=0,
        mortgage_term=30,
        down_payment_pct=0.20,
    )
    fin = result['financials']

    occupied_nights = 0.5 * 365
    turnovers = occupied_nights / 3.2
    operating = (
        40000 * 0.03
        + turnovers * 125
        + turnovers * 25
        + 275 * 12 * (0.6 + 0.4 * 0.5)
        + (75 + 50) * 12
        + 200000 * 0.01
    )
    noi = 40000 - operating
    mortgage = 160000 / 30
    debt_service = mortgage + 200000 * 0.018 + 200000 * 0.007
    cashflow = noi - debt_service

    assert fin['noi'] == pytest.approx(round(noi, 2))
    assert fin['cap_rate'] == pytest.approx(round(noi / 200000 * 100, 2))
    assert fin['roi'] == pytest.approx(round(cashflow / 40000 * 100, 2))
    assert fin['dscr'] == pytest.approx(round(noi / debt_service, 2))
    assert fin['annual_cashflow'] == pytest.approx(round(cashflow, 2))
    assert fin['cap_rate'] == pytest.approx(12.05, abs=0.01)
    assert fin['roi'] == pytest.approx(34.43, abs=0.02)
    assert fin['dscr'] == pytest.approx(2.33, abs=0.01)


def test_dscr_below_one_when_the_mortgage_is_larger_than_income():
    result = score_property(
        900000,
        20000,
        0.3,
        city_key='austin',
        mortgage_rate=0.08,
        down_payment_pct=0.10,
    )
    fin = result['financials']
    assert fin['dscr'] < 1
    assert fin['annual_cashflow'] < 0
    assert fin['cap_rate'] == pytest.approx(
        round(fin['noi'] / fin['purchase_price'] * 100, 2)
    )
    assert fin['roi'] == pytest.approx(
        round(fin['annual_cashflow'] / fin['down_payment'] * 100, 2)
    )


def test_standard_rate_mortgage_matches_the_published_payment():
    # $200,000 at 6% for 30 years is $1,199.10 a month.
    annual = calculate_annual_mortgage(200000, rate=0.06, years=30)
    assert annual / 12 == pytest.approx(1199.10, abs=0.05)
