"""
STR Investment Score (0-100) composite scoring.

Components:
  - Cap rate (NOI / purchase price) — 0-30 pts
  - ROI (annual cashflow / down payment) — 0-30 pts
  - Revenue-to-price ratio — 0-25 pts
  - Predicted occupancy — 0-15 pts
"""
import numpy as np


# Assumptions for financial modeling
EXPENSE_RATIO = 0.35          # 35% of revenue goes to expenses
MORTGAGE_RATE = 0.07          # 7% annual interest
MORTGAGE_TERM_YEARS = 30
DOWN_PAYMENT_PCT = 0.20       # 20% down
PROPERTY_TAX_RATE = 0.018     # 1.8% of purchase price (Austin TX)
INSURANCE_ANNUAL = 2000


def calculate_annual_mortgage(loan_amount, rate=MORTGAGE_RATE, years=MORTGAGE_TERM_YEARS):
    monthly_rate = rate / 12
    n_payments = years * 12
    if monthly_rate == 0:
        return loan_amount / n_payments * 12
    monthly = loan_amount * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
    return monthly * 12


def score_property(purchase_price, predicted_annual_revenue, predicted_occupancy):
    """
    Calculate composite STR Investment Score (0-100).

    Returns dict with score breakdown and financial metrics.
    """
    if purchase_price <= 0:
        return {'score': 0, 'breakdown': {}, 'financials': {}}

    # Financial calculations
    gross_revenue = predicted_annual_revenue
    operating_expenses = gross_revenue * EXPENSE_RATIO
    noi = gross_revenue - operating_expenses

    down_payment = purchase_price * DOWN_PAYMENT_PCT
    loan_amount = purchase_price - down_payment
    annual_mortgage = calculate_annual_mortgage(loan_amount)
    annual_taxes = purchase_price * PROPERTY_TAX_RATE
    annual_insurance = INSURANCE_ANNUAL

    annual_cashflow = noi - annual_mortgage - annual_taxes - annual_insurance

    # Scoring components (benchmarks calibrated for Austin TX STR market)
    cap_rate = noi / purchase_price
    cap_rate_score = min(30, max(0, (cap_rate / 0.10) * 30))

    roi = annual_cashflow / down_payment if down_payment > 0 else 0
    roi_score = min(30, max(0, ((roi + 0.1) / 0.3) * 30))

    rev_to_price = gross_revenue / purchase_price
    rev_to_price_score = min(25, max(0, (rev_to_price / 0.12) * 25))

    occ_score = min(15, max(0, (predicted_occupancy / 0.75) * 15))

    total_score = round(cap_rate_score + roi_score + rev_to_price_score + occ_score, 1)
    total_score = min(100, max(0, total_score))

    return {
        'score': total_score,
        'breakdown': {
            'cap_rate_score': round(cap_rate_score, 1),
            'roi_score': round(roi_score, 1),
            'revenue_price_score': round(rev_to_price_score, 1),
            'occupancy_score': round(occ_score, 1),
        },
        'financials': {
            'purchase_price': purchase_price,
            'down_payment': round(down_payment, 2),
            'annual_revenue': round(gross_revenue, 2),
            'operating_expenses': round(operating_expenses, 2),
            'noi': round(noi, 2),
            'annual_mortgage': round(annual_mortgage, 2),
            'annual_taxes': round(annual_taxes, 2),
            'annual_insurance': INSURANCE_ANNUAL,
            'annual_cashflow': round(annual_cashflow, 2),
            'monthly_cashflow': round(annual_cashflow / 12, 2),
            'cap_rate': round(cap_rate * 100, 2),
            'roi': round(roi * 100, 2),
            'revenue_to_price': round(rev_to_price * 100, 2),
        }
    }
