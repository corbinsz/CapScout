"""
STR Investment Score (0-100) composite scoring.

Based on vetted real estate investment principles from AirDNA, BiggerPockets,
Mashvisor, and industry-standard STR underwriting metrics.

Components (research-backed weights):
  - Gross Yield (annual revenue / price)       — 0-25 pts  (#1 quick-screen metric)
  - Cash-on-cash ROI (cashflow / down payment) — 0-25 pts  (#2 leveraged return metric)
  - Cap rate (NOI / price)                     — 0-20 pts  (unleveraged profitability)
  - DSCR (NOI / total debt service)            — 0-20 pts  (debt coverage safety)
  - Predicted occupancy                        — 0-10 pts  (demand indicator)

Score interpretation:
  75+ = Excellent (strong returns, solid debt coverage)
  55-74 = Good (profitable or near breakeven with upside)
  40-54 = Fair (marginal deal, watch closely)
  <40 = Poor (unlikely to perform as STR investment)

Thresholds calibrated for 7%+ interest rate environment per 2024-2025
STR investor community standards (BiggerPockets, AirDNA, Rabbu).
"""
from cities import get_city_or_default


# Assumptions for financial modeling
MORTGAGE_RATE = 0.07          # 7% annual interest
MORTGAGE_TERM_YEARS = 30
DOWN_PAYMENT_PCT = 0.20       # 20% down
# Fallbacks if city config is missing a field
DEFAULT_PROPERTY_TAX_RATE = 0.018
DEFAULT_INSURANCE_RATE = 0.005
DEFAULT_EXPENSE_RATIO = 0.35
DEFAULT_STR_PERMIT_ANNUAL = 0


def calculate_annual_mortgage(loan_amount, rate=MORTGAGE_RATE, years=MORTGAGE_TERM_YEARS):
    monthly_rate = rate / 12
    n_payments = years * 12
    if monthly_rate == 0:
        return loan_amount / n_payments * 12
    monthly = loan_amount * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
    return monthly * 12


def score_property(purchase_price, predicted_annual_revenue, predicted_occupancy,
                   city_key=None, mortgage_rate=None, mortgage_term=None,
                   down_payment_pct=None, hoa_annual=None):
    """
    Calculate composite STR Investment Score (0-100).

    Returns dict with score breakdown and financial metrics.
    Optional overrides for mortgage_rate, mortgage_term, down_payment_pct
    let callers customize financial assumptions.
    hoa_annual: known annual HOA fees (pass hoa_monthly * 12).
    """
    if purchase_price <= 0:
        return {'score': 0, 'breakdown': {}, 'financials': {}}

    city = get_city_or_default(city_key)
    tax_rate = city.get('property_tax_rate', DEFAULT_PROPERTY_TAX_RATE)
    insurance_rate = city.get('insurance_rate', DEFAULT_INSURANCE_RATE)
    expense_ratio = city.get('expense_ratio', DEFAULT_EXPENSE_RATIO)
    str_permit = city.get('str_permit_annual', DEFAULT_STR_PERMIT_ANNUAL)

    # Use overrides if provided, else fall back to module constants
    eff_mortgage_rate = mortgage_rate if mortgage_rate is not None else MORTGAGE_RATE
    eff_mortgage_term = mortgage_term if mortgage_term is not None else MORTGAGE_TERM_YEARS
    eff_down_pct = down_payment_pct if down_payment_pct is not None else DOWN_PAYMENT_PCT
    eff_hoa_annual = hoa_annual if hoa_annual and hoa_annual > 0 else 0

    # Financial calculations
    gross_revenue = predicted_annual_revenue
    operating_expenses = gross_revenue * expense_ratio
    noi = gross_revenue - operating_expenses

    down_payment = purchase_price * eff_down_pct
    loan_amount = purchase_price - down_payment
    annual_mortgage = calculate_annual_mortgage(loan_amount, rate=eff_mortgage_rate, years=eff_mortgage_term)
    annual_taxes = purchase_price * tax_rate
    annual_insurance = purchase_price * insurance_rate

    # Total debt service (PITIA + HOA + permits)
    total_debt_service = annual_mortgage + annual_taxes + annual_insurance + str_permit + eff_hoa_annual
    annual_cashflow = noi - total_debt_service

    # Key ratios
    gross_yield = gross_revenue / purchase_price
    cap_rate = noi / purchase_price
    roi = annual_cashflow / down_payment if down_payment > 0 else 0
    dscr = noi / total_debt_service if total_debt_service > 0 else 0

    # =====================================================================
    # SCORING COMPONENTS — thresholds from industry research
    # =====================================================================

    # 1. Gross Yield / Revenue-to-Price (0-25 pts)
    # THE #1 quick-screen metric per BiggerPockets & AirDNA.
    # 10% = market baseline (listed homes), 15% = good deal, 20%+ = excellent.
    yield_score = min(25, max(0, (gross_yield / 0.20) * 25))

    # 2. Cash-on-Cash ROI (0-25 pts)
    # The primary "deal or no deal" metric for leveraged investors.
    # 8% = good, 15% = excellent, 20%+ = exceptional.
    # Negative ROI still gets partial credit (property may appreciate).
    if roi >= 0:
        roi_score = min(25, 5 + (roi / 0.20) * 20)
    else:
        # -10% or worse = 0 pts, 0% = 5 pts
        roi_score = max(0, 5 + (roi / 0.10) * 5)

    # 3. Cap Rate (0-20 pts)
    # STR cap rates run 1.5-2x higher than LTR.
    # 5% = decent, 8% = good, 10%+ = excellent.
    cap_rate_score = min(20, max(0, (cap_rate / 0.10) * 20))

    # 4. DSCR — Debt Service Coverage Ratio (0-20 pts)
    # Critical for leveraged purchases. Lenders want 1.25+.
    # <0.75 = 0 pts, 1.0 = breakeven (10 pts), 1.25 = strong (16 pts), 1.5+ = 20 pts.
    if dscr >= 1.0:
        dscr_score = min(20, 10 + ((dscr - 1.0) / 0.5) * 10)
    else:
        dscr_score = max(0, (dscr / 1.0) * 10)

    # 5. Occupancy (0-10 pts)
    # National avg ~55%. Good = 55-65%, strong = 65-75%.
    occ_score = min(10, max(0, (predicted_occupancy / 0.75) * 10))

    total_score = yield_score + roi_score + cap_rate_score + dscr_score + occ_score
    total_score = round(min(100, max(0, total_score)), 1)

    return {
        'score': total_score,
        'breakdown': {
            'yield_score': round(yield_score, 1),
            'roi_score': round(roi_score, 1),
            'cap_rate_score': round(cap_rate_score, 1),
            'dscr_score': round(dscr_score, 1),
            'occupancy_score': round(occ_score, 1),
        },
        'financials': {
            'purchase_price': purchase_price,
            'down_payment': round(down_payment, 2),
            'annual_revenue': round(gross_revenue, 2),
            'operating_expenses': round(operating_expenses, 2),
            'expense_ratio': round(expense_ratio * 100, 1),
            'noi': round(noi, 2),
            'annual_mortgage': round(annual_mortgage, 2),
            'annual_taxes': round(annual_taxes, 2),
            'tax_rate': round(tax_rate * 100, 2),
            'annual_insurance': round(annual_insurance, 2),
            'insurance_rate': round(insurance_rate * 100, 2),
            'str_permit_annual': round(str_permit, 2),
            'hoa_annual': round(eff_hoa_annual, 2),
            'annual_cashflow': round(annual_cashflow, 2),
            'monthly_cashflow': round(annual_cashflow / 12, 2),
            'cap_rate': round(cap_rate * 100, 2),
            'roi': round(roi * 100, 2),
            'revenue_to_price': round(gross_yield * 100, 2),
            'dscr': round(dscr, 2),
            'mortgage_rate': round(eff_mortgage_rate * 100, 2),
            'mortgage_term': eff_mortgage_term,
            'down_payment_pct': round(eff_down_pct * 100, 2),
        }
    }
