"""City grades from predicted annual revenue, and the gross-yield ratio.

The cuts are percentiles of estimated revenue on the training listings.
They are not a claim about profit. Purchase price, if the user types one,
is used only for the gross ratio.
"""

from __future__ import annotations

# Cuts of estimated annual revenue inside the city. Training rows only.
# A is the top fifth, F is the bottom fifth. Not fit to a test metric.
GRADE_ORDER = (
    ("A", "Strong", "p80"),
    ("B", "Above average", "p60"),
    ("C", "Middle", "p40"),
    ("D", "Below average", "p20"),
    ("F", "Weak", None),
)


def grade_revenue(revenue: float, cuts: dict[str, float]) -> dict[str, str | float]:
    """Letter and plain words for one predicted annual revenue.

    ``cuts`` has ``p20``, ``p40``, ``p60``, and ``p80`` in dollars. A value
    on a boundary takes the higher grade.
    """
    value = float(revenue)
    letter = "F"
    words = "Weak"
    for candidate, name, key in GRADE_ORDER:
        if key is None:
            letter, words = candidate, name
            break
        if value >= float(cuts[key]):
            letter, words = candidate, name
            break
    return {"letter": letter, "words": words, "revenue": value}


def annual_revenue(nightly_rate: float, booked_nights: float) -> float:
    """Predicted gross: rate times nights. No fees and no vacancy haircut."""
    return float(nightly_rate) * float(booked_nights)


def gross_yield(revenue: float, purchase_price: float | None) -> float | None:
    """Revenue divided by a price the user typed. None when there is no price."""
    if purchase_price is None:
        return None
    price = float(purchase_price)
    if price <= 0:
        return None
    return float(revenue) / price
