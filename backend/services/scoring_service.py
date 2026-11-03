"""
STR Investment Score (0-100) composite scoring.

Based on vetted real estate investment principles from AirDNA, BiggerPockets,
Mashvisor, and industry-standard STR underwriting metrics.

Components (research-backed weights):
  - Gross Yield (annual revenue / price)       -- 0-25 pts  (#1 quick-screen metric)
  - Cash-on-cash ROI (cashflow / down payment) -- 0-25 pts  (#2 leveraged return metric)
  - Cap rate (NOI / price)                     -- 0-20 pts  (unleveraged profitability)
  - DSCR (NOI / total debt service)            -- 0-20 pts  (debt coverage safety)
  - Predicted occupancy                        -- 0-10 pts  (demand indicator)

Score interpretation:
  70+ = Excellent (strong returns, solid debt coverage)
  55-69 = Good (profitable or near breakeven with upside)
  45-54 = Fair (marginal deal, watch closely)
  <45 = Poor (unlikely to perform as STR investment)

Expense model: Itemized by category instead of a flat percentage.
  - Platform fees (3% of revenue)
  - Cleaning costs (per-turnover, city-specific)
  - Supplies & consumables (per-turnover)
  - Utilities (city-specific, partly occupancy-driven)
  - Maintenance reserve (1% of property value)
  - WiFi + miscellaneous (fixed monthly)
"""
from cities import get_city_or_default


# Assumptions for financial modeling
MORTGAGE_RATE = 0.07          # 7% annual interest
MORTGAGE_TERM_YEARS = 30
DOWN_PAYMENT_PCT = 0.20       # 20% down

# Fallbacks if city config is missing a field
DEFAULT_PROPERTY_TAX_RATE = 0.018
DEFAULT_INSURANCE_RATE = 0.005
DEFAULT_STR_PERMIT_ANNUAL = 0

# Itemized expense constants (same for all cities)
PLATFORM_FEE_RATE = 0.03       # Airbnb host-only fee (3% of revenue)
SUPPLIES_PER_TURNOVER = 25     # Toiletries, coffee, linens per stay
MAINTENANCE_RATE = 0.01        # 1% of property value annually (industry standard)
WIFI_MONTHLY = 75              # Internet service
MISC_MONTHLY = 50              # Pest control, software, misc subscriptions

# Fallbacks for city expense fields
DEFAULT_CLEANING_PER_TURNOVER = 120
DEFAULT_AVG_STAY_NIGHTS = 3.2
DEFAULT_BASE_UTILITIES_MONTHLY = 250


def calculate_annual_mortgage(loan_amount, rate=MORTGAGE_RATE, years=MORTGAGE_TERM_YEARS):
    monthly_rate = rate / 12
    n_payments = years * 12
    if monthly_rate == 0:
        return loan_amount / n_payments * 12
    monthly = loan_amount * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
    return monthly * 12


def calculate_operating_expenses(gross_revenue, predicted_occupancy, purchase_price, city):
    """Calculate itemized operating expenses.

    Returns (total_opex, breakdown_dict).
    """
    cleaning_cost = city.get('cleaning_per_turnover', DEFAULT_CLEANING_PER_TURNOVER)
    avg_stay = city.get('avg_stay_nights', DEFAULT_AVG_STAY_NIGHTS)
    base_util = city.get('base_utilities_monthly', DEFAULT_BASE_UTILITIES_MONTHLY)

    # Turnovers per year: occupied nights / avg stay length
    occupied_nights = predicted_occupancy * 365
    turnovers = occupied_nights / avg_stay if avg_stay > 0 else 0

    # Revenue-based
    platform_fees = gross_revenue * PLATFORM_FEE_RATE

    # Per-turnover (scales with occupancy)
    cleaning = turnovers * cleaning_cost
    supplies = turnovers * SUPPLIES_PER_TURNOVER

    # Utilities: base (60%) always on + variable (40%) scales with occupancy
    annual_utilities = base_util * 12 * (0.6 + 0.4 * predicted_occupancy)

    # Fixed monthly
    annual_fixed = (WIFI_MONTHLY + MISC_MONTHLY) * 12

    # Property-value based
    maintenance = purchase_price * MAINTENANCE_RATE

    total = platform_fees + cleaning + supplies + annual_utilities + annual_fixed + maintenance

    breakdown = {
        'platform_fees': round(platform_fees, 2),
        'cleaning': round(cleaning, 2),
        'supplies': round(supplies, 2),
        'utilities': round(annual_utilities, 2),
        'fixed_costs': round(annual_fixed, 2),
        'maintenance': round(maintenance, 2),
        'turnovers': round(turnovers, 1),
        'cleaning_per_turnover': cleaning_cost,
        'avg_stay_nights': avg_stay,
        'base_utilities_monthly': base_util,
    }

    return round(total, 2), breakdown


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
    str_permit = city.get('str_permit_annual', DEFAULT_STR_PERMIT_ANNUAL)

    # Use overrides if provided, else fall back to module constants
    eff_mortgage_rate = mortgage_rate if mortgage_rate is not None else MORTGAGE_RATE
    eff_mortgage_term = mortgage_term if mortgage_term is not None else MORTGAGE_TERM_YEARS
    eff_down_pct = down_payment_pct if down_payment_pct is not None else DOWN_PAYMENT_PCT
    eff_hoa_annual = hoa_annual if hoa_annual and hoa_annual > 0 else 0

    # Financial calculations
    gross_revenue = predicted_annual_revenue

    # Itemized operating expenses
    operating_expenses, opex_breakdown = calculate_operating_expenses(
        gross_revenue, predicted_occupancy, purchase_price, city)

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
    # SCORING COMPONENTS -- thresholds from industry research
    # =====================================================================

    # 1. Gross Yield / Revenue-to-Price (0-25 pts)
    yield_score = min(25, max(0, (gross_yield / 0.20) * 25))

    # 2. Cash-on-Cash ROI (0-25 pts)
    if roi >= 0:
        roi_score = min(25, 5 + (roi / 0.20) * 20)
    else:
        roi_score = max(0, 5 + (roi / 0.10) * 5)

    # 3. Cap Rate (0-20 pts)
    cap_rate_score = min(20, max(0, (cap_rate / 0.10) * 20))

    # 4. DSCR (0-20 pts)
    if dscr >= 1.0:
        dscr_score = min(20, 10 + ((dscr - 1.0) / 0.5) * 10)
    else:
        dscr_score = max(0, (dscr / 1.0) * 10)

    # 5. Occupancy (0-10 pts)
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
            'opex': opex_breakdown,
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


def explain_deal(score, financials, confidence_level=None):
    """One plain-English sentence a buyer can read without a spreadsheet."""
    if score is None or not financials:
        return 'This home is not scored yet. A trained model and a list price are both required.'
    if score >= 70:
        label = 'Excellent'
    elif score >= 55:
        label = 'Good'
    elif score >= 45:
        label = 'Fair'
    else:
        label = 'Weak'
    cashflow = financials.get('monthly_cashflow') or 0
    if cashflow >= 0:
        cash = f'about ${cashflow:,.0f} a month left after the mortgage, taxes, insurance, and operating costs'
    else:
        cash = f'about ${abs(cashflow):,.0f} a month short of covering the mortgage, taxes, insurance, and operating costs'
    cap = financials.get('cap_rate')
    dscr = financials.get('dscr')
    extras = []
    if cap is not None:
        extras.append(f'{cap}% cap rate')
    if dscr is not None:
        extras.append(f'debt coverage {dscr}')
    extra = f" ({', '.join(extras)})" if extras else ''
    confidence = ''
    if confidence_level == 'low':
        confidence = ' Confidence is low because few short-term rentals in the training data sit nearby.'
    elif confidence_level == 'high':
        confidence = ' Confidence is higher because many short-term rentals nearby were in the training data.'
    return (
        f'{label} under the loan assumptions on this page: {cash}{extra}.{confidence} '
        'This is a model estimate, not financial advice.'
    )
