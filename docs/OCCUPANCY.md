# How the nights estimate was chosen

CapScout already predicts an advertised nightly rate. This note is the reading behind a second number: how many nights a home would likely book in a year, and a plain grade of the gross those two numbers imply. It is a personal project. The measurement is in [reports/FINDINGS.md](../reports/FINDINGS.md). On the seed-31 holdout the nights model did not clear the bar written down in that note.

## What public data can and cannot see

Airbnb's calendar does not say whether a blocked night was booked or simply turned off. Inside Airbnb says this directly, and this project already refused to treat `availability_365` as occupancy. AirDNA-style products try to separate those two with scraped calendars and a private model. That data is not free and not in this repo, so it is not the target.

The public alternative is Inside Airbnb's San Francisco model, named for the occupancy work the San Francisco Planning Department and the Budget and Legislative Analyst's Office did around 2015. The published version, on Inside Airbnb's data-assumptions page, is:

- Turn reviews into bookings with a **50% review rate**. A 72% rate has been attributed to Airbnb and is not something an outsider can check. A 30.5% rate came from comparing public reviews with the New York Attorney General's 2014 Airbnb report. Inside Airbnb sits in the middle, at 50%, and says the lower rate is probably closer to the truth but not conservative enough once deleted listings are considered.
- Multiply bookings by an **average length of stay**. Where a city has a public statement of that average, they use it (San Francisco's often-cited figure is 5.5 nights). Where it does not, they use **3 nights**. If the listing's minimum stay is longer, they use the minimum instead.
- **Cap occupancy at 70%**, so a listing with an unusually high review rate cannot look like a full hotel.

Academic and planning work that stays on free data uses this same review proxy, or a close cousin of it. It is an estimate of review activity scaled into nights. It is not a count of stays. Guests who never review are invisible, and a host who asks for reviews looks busier than one who does not.

## The target this study actually uses

Estimated booked nights in the last 12 months, for Austin, San Diego, and Seattle:

`nights = min(0.70 × 365, (number_of_reviews_ltm / 0.50) × stay)`

`number_of_reviews_ltm` is the review count Inside Airbnb publishes for the trailing year. A lifetime "reviews per month" rate mixes old years into the label, so it is not used. Stay length is **3 nights for every city**. I did not find a primary-source statement of average stay for these three cities that I was willing to treat as a measurement, and the hand-set 3.2 / 4.0 / 3.0 figures already in `data/cities.py` are comments, not data. Using one stated number keeps a guess from becoming a city effect inside the label. If minimum nights is greater than 3 and at most 90, that minimum replaces the 3. The 90-night ceiling is the same guard the monthly proxy already uses: a 365-night "minimum" is not a stay length. The cap is **255.5 nights**, which is 70% of 365.

A listing with no reviews in the last 12 months is left out. It is not coded as zero. A new or dormant ad has not been observed. The model therefore answers a narrower question: among listings that received at least one review in the last year, how many nights does the review proxy imply, and how much of that can a model see from the house itself?

Minimum nights is part of the label and is **not** a feature. Two similar houses with different house rules get different labels, and the model is not allowed to read the rule that created the difference. That noise is the cost of following the published formula without leaking it.

## What the model is allowed to see

The use is a home on the market, not an existing ad. The feature ladder, simplest first, is size and room type, then coordinates and distance to downtown and the airport, then census tract income, rent, home value, and density, then building type, then amenities and beds. The earliest prefix within 1% of the best development MAE is kept. The same 1% rule decides whether an unlogged target replaces log1p, and whether an equal CatBoost and XGBoost blend replaces XGBoost. Depth stays at 4, the published price-model depth. Review counts, review scores, availability, host performance, minimum nights, and price are not candidates. A one-seed fit that adds `number_of_reviews_ltm` is reported only to show that the column reconstructs the label.

The price model is unchanged. Its published numbers are the seed-23 table. For the revenue error only, that same price specification is refit with the nights holdout, the host purge, and the 1 km buffer removed from training, then multiplied by the nights prediction. The saved price artifact is not replaced. A person scoring a new address is not in that training set either.

## The split and the bar

Seed 31 holds out 20% of the 0.03-degree cells on the listings that have a nights label. It is not seed 0, 1, 2, 3, 4, 7, 11, 23, or 42, and it was not searched. The ids were written before any occupancy model was fit. Hosts and a 1 km buffer are removed from training, as in the price study. Development seeds 0 through 4 choose the prefix. The holdout is scored once.

The bar was written down first. A city clears it only when holdout MAE is at most 90% of the neighbourhood-median MAE **and** holdout R² is at least 0.10. A low / medium / high band, cut at the training tertiles, clears its own bar only when accuracy beats the neighbourhood's majority class by 8 points and balanced accuracy is higher too. Missing either bar is the result. It is not a reason to edit the target.

## The grade

Estimated annual revenue is predicted nightly rate times predicted nights. Inside each city, the training listings' estimated revenue (advertised rate times the review-based nights) is cut at the 20th, 40th, 60th, and 80th percentiles:

| Grade | Words | Predicted revenue |
| --- | --- | --- |
| A | Strong | at or above the 80th percentile |
| B | Above average | at or above the 60th, below the 80th |
| C | Typical | at or above the 40th, below the 60th |
| D | Below average | at or above the 20th, below the 40th |
| F | Weak | below the 20th percentile |

A value on a boundary takes the higher grade. This is a rank of gross revenue among reviewed listings in that city. It does not know the purchase price, the mortgage, taxes, insurance, cleaning, platform fees, or whether a short-term rental is allowed on the block. If someone types a purchase price, the app divides predicted revenue by that price and labels the result a rough gross yield. That division is not a cap rate.

## Sources

- Inside Airbnb, Data assumptions, "The Occupancy Model." The 50% review rate, the 3-night fallback, the minimum-stay rule, and the 70% cap. https://insideairbnb.com/data-assumptions/
- San Francisco Budget and Legislative Analyst's Office, analysis of short-term rentals (the 72% and 30.5% review rates Inside Airbnb discusses).
- Alex Marqusee, San Francisco Planning Department, the descriptive memo the San Francisco model is named for.
- This repository's earlier note in `data/occupancy.py`, which implemented the monthly version of the same proxy and left listings with no reviews unobserved.

## What would replace this label

A daily calendar panel is now collected for a fixed sample of these three cities. The rules that turn repeated snapshots into a booked-night rate, and the 26 weeks to wait before refitting, are in [CALENDAR.md](CALENDAR.md). This review proxy stays the scored target until that panel exists and a fresh spatial holdout is frozen.
