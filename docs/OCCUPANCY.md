# Occupancy

The first models learned `1 - availability_30 / 30`. On a holdout that target had an error of about 0.26 to 0.30, and a model barely beat predicting the median (a ratio of about 0.96). The calendar and a review-based estimate of the same homes correlate at about 0.05. They are not measuring the same thing.

## What was compared

Both targets were fit with the same XGBoost settings on a 20% holdout of whole listings, in Austin, Seattle, Asheville, and Nashville.

| Target | What it is | Mean error ratio vs predicting the median |
| --- | --- | --- |
| Calendar | Next 30 nights marked unavailable | 0.96 |
| San Francisco review model | Reviews last year / 0.5, times the stay, / 365, cap 70% | 0.89 |

The review model is the one in production. Its holdout error is about 0.16 to 0.22 (16 to 22 percentage points). R-squared is still modest, from about 0.09 in Austin to about 0.29 in Hawaii. Location and size do not determine how hard a host works.

## The review formula

Documented by Inside Airbnb as the San Francisco model (https://insideairbnb.com/data-assumptions/):

- 50% of stays are assumed to leave a review.
- Bookings = reviews in the last 12 months / 0.5.
- Nights = bookings times the length of stay.
- Length of stay is the city's average (`avg_stay_nights` in `cities.py`). If the listing's minimum is longer, that minimum is used, but not past 14 nights. A 30-night minimum is a monthly rental, and using it raw would turn two reviews into a fake full calendar.
- Occupancy = nights / 365, and it cannot exceed 70%.

## Calendar patterns, without the raw calendar file

The repo has each listing's `availability_30` and `availability_365`, which are the calendar summaries. It does not have `calendar.csv.gz` (one row per listing per day). Those files are large, and a night marked unavailable is still either a booking or a block.

On the four benchmark cities, about 3% to 6% of listings have the next month fully closed and no review in the last year. About 3% have the year mostly open, the next month closed, and no reviews. The old target called both of those "occupied."

## What this does not fix

A for-sale house has no reviews yet. The model predicts the review-based rate of similar homes nearby. It is a typical-operator estimate, not a promise for the house you are buying.
