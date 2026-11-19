CapScout is a personal project built in Corbin's spare time, for learning and as a portfolio piece. It is not a commercial product.

# CapScout

How close can a public model get to an Airbnb listing's advertised nightly rate, without training on the listing next door or on the same host?

The long version is [reports/FINDINGS.md](reports/FINDINGS.md). The reading behind the nightly rate is [docs/RESEARCH.md](docs/RESEARCH.md). The reading behind booked nights is [docs/OCCUPANCY.md](docs/OCCUPANCY.md). Sources and licenses are in [DATA.md](DATA.md).

## The result

One new spatial holdout per city, seed 23, frozen before this round fit anything and scored once at the end. The point model is the v3 blend of CatBoost and XGBoost on log price. Nothing tried this round — earlier snapshots, matched comps, a sentence embedding, tail weights, LightGBM, a small hyperparameter grid, a weighted blend, smearing, or an isotonic correction — cleared a pre-set 1% bar on development MAE. The only change that did clear its bar is the 80% interval, which is now fit separately in bands of the predicted price.

| City | Test R² | Test MAE | Test RMSE | MAPE | Median APE | 80% coverage | Listings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin | 0.604 | $61.85 | $125.89 | 33.3% | 23.6% | 81.4% | 974 |
| San Diego | 0.545 | $56.64 | $135.34 | 26.7% | 19.4% | 80.1% | 391 |
| Seattle | 0.667 | $33.92 | $50.98 | 25.2% | 21.5% | 78.9% | 266 |

These cells are not v3's cells. On the same rows, the v3 specification matches the table exactly. Seed 23 was an easier dollar problem, and a smaller one. San Diego's holdout is 391 listings. Seattle's is 266. Do not read $61.85 as an improvement on v3's $83.95.

Against a neighbourhood-by-room-type median on these same rows, the model removes 37.5% of the error in Austin, 46.1% in San Diego, and 36.4% in Seattle. The baseline MAE is $98.98, $105.07, and $53.31. That baseline is weak on a spatial split. The percent is the gap to it.

Seed 7 had already been published, so it was not used to choose anything. On those rows the dollars match v3 exactly: R² 0.451 / 0.675 / 0.526 and MAE $83.95 / $99.52 / $38.75. Coverage moved to 74.9% / 76.8% / 80.4%, from v3's 73.2% / 77.1% / 83.2%. Austin is still short of 80%. Against seed 7's neighbourhood median the model removes 34.2%, 46.8%, and 37.4% of the error, which is v3's gap, because the predictions are v3's.

Development R², on folds that exclude all three holdouts, was 0.547 ± 0.062 in Austin, 0.678 ± 0.075 in San Diego, and 0.454 ± 0.102 in Seattle.

An older pipeline trained on area-median prices and showed R² near 0.97. That was the target being easy. Scored against the listing's own price, the same exercise is about R² 0.385. It is a demonstration, not a result of this study.

## Booked nights

A separate question, with its own holdout: how many nights in a year would this home book?

The label is not a booking count. It is Inside Airbnb's review-rate estimate for the last 12 months: reviews in that year divided by 0.50, times a stated stay of 3 nights (or the listing's minimum, when that minimum is longer and at most 90), capped at 255.5 nights. Listings with no reviews in the last year are left out, not coded as zero. The model is not allowed to see the review count, the minimum stay, availability, or the price. Those either build the label or are not known for a house that has never been listed.

Seed 31 holds out 20% of the spatial cells. It is not 7, 11, or 23, and it was frozen before any nights model was fit. The bar, written down first, is a 10% lower average miss than the neighbourhood median **and** an R² of at least 0.10. The model did not clear it.

| City | Test R² | Test MAE | Neighbourhood median MAE | Error removed | Revenue R² | Revenue MAE | Listings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin | 0.073 | 70.0 nights | 76.9 nights | 8.9% | 0.133 | $12,906 | 2,420 |
| San Diego | 0.098 | 72.0 nights | 78.9 nights | 8.7% | -0.034 | $18,720 | 2,655 |
| Seattle | 0.078 | 71.6 nights | 79.5 nights | 9.9% | 0.117 | $10,470 | 765 |

The nights model is an XGBoost on the house itself: size, room type, location, census, building type, and amenities. Depth 4, the same depth as the price model. It trims about 9% off the neighbourhood median and leaves most of the variation unexplained. That is short of the bar. San Diego's revenue number, predicted rate times predicted nights, has a negative R². The nights error swamps the rate.

A low / medium / high band, cut at the training tertiles, beat the neighbourhood's usual band by about 10 points in Austin (43.1% vs 32.6%) and San Diego (43.0% vs 32.7%). It did not in Seattle (41.2% vs 40.1%). The middle band's predicted probability runs ahead of how often that band is actually right. It is not a calibrated occupancy forecast.

The grade in the app is a rank of predicted annual revenue inside the city: A at or above the 80th percentile of training estimated revenue, then B, C, and D at the 60th, 40th, and 20th, and F below that. Austin's cuts are $3,240, $7,560, $14,400, and $25,182. San Diego's are $7,252.80, $15,590.60, $26,346, and $43,024.80. Seattle's are $6,402, $13,020, $19,929, and $29,104.20. On the holdout the letter matched the grade of the review-based revenue 30.6% of the time in Austin, 30.5% in San Diego, and 25.0% in Seattle. The grade does not know the purchase price, the mortgage, taxes, or whether a short-term rental is allowed. A purchase price typed into the app is only a rough gross yield, labeled as such.

## Calendar panel

The nights model is weak because the label is a review proxy. A daily job now watches a fixed panel of real calendars so a later model can train on nights that actually closed. The rules and the limits are in [docs/CALENDAR.md](docs/CALENDAR.md). The snapshots go to the `calendar-data` branch, compressed, so main stays as the study.

What is collected: 300 listings each in Austin, San Diego, and Seattle, drawn once from the Inside Airbnb files already in the repo (seed 17, stratified by neighbourhood and room type). The ids are `data/calendar_sample.json`. Once a day the job reads each listing's public availability calendar and appends `listing_id`, `date`, `available`, `min_nights`, `price` when the calendar includes one, and `fetched_at`. It waits about 2 seconds between listings, retries ordinary server errors with backoff, names itself in the user agent, and stops if the response is a block or a challenge page. There is no login, no captcha handling, and no proxy. If that public calendar cannot be read, the job stores Inside Airbnb's published `calendar.csv` for the same ids instead and says so in the run report. One published file cannot show a night flip, so that fallback is a supply snapshot, not a booked-night rate.

How a booking is inferred, once two or more snapshots exist:

- A night that was available and then unavailable, in a run of 1 to 13 nights, counts as booked when we still saw it open inside 45 days of the night and the run is at least as long as the minimum stay that was posted.
- A run of 14 nights or more that was already closed when the first night was still 90 days out counts as an owner block. Ninety days is the shortest availability window a host can set.
- A stretch of 21 unavailable nights through the last date in the snapshot, never seen open, counts as that availability window, not as demand.
- Half or more of the open calendar closing between two snapshots at most 3 days apart counts as one owner block.
- A night that was already unavailable the first time we saw it stays out of the booked count. A night that opens again before the date is not a booking. A night with no snapshot within 2 days of itself stays unresolved.

The booked-night rate is booked nights divided by booked plus nights that were still open at the date. Blocks, the closed window, censored nights, and unresolved nights are reported and left out of the rate. A one-night owner hold can look like a one-night booking when the minimum stay is 1. A real stay of 14 nights or a stay booked more than 45 days ahead is left as uncertain. The price column is kept and is not part of the rule, because the public calendar often omits it.

Retrain after 26 weeks of daily snapshots. Nights already closed on day one are censored, and a night we first see far out does not become a resolved outcome until that date arrives, about 13 weeks later for the far cutoff. A rate near 30% then needs on the order of 80 offered nights for a standard error near 5 points, which is about 12 weeks more. 13 + 12 is 25, and a half year also keeps the window off a single holiday peak, so the wait is 26 weeks. That label is the rate over the tracked window. It is not an annual night count. An annual label waits for 52 weeks. When the 26 weeks exist, occupancy accuracy will be re-tested on a fresh spatial holdout: a new seed, not seed 31, with the cell ids frozen before the fit, scored once.

```bash
python scripts/collect_calendars.py --out-dir calendars
python scripts/build_occupancy_labels.py --snapshots calendars --out reports/calendar_occupancy.csv
```

The GitHub Actions workflow runs the collector daily at 13:17 UTC and on demand. Local snapshot files are gitignored. The committed panel is the sample json, not the daily files.

## What did not help the nightly rate

Matched comps within 15 km, an eight-component sentence embedding, earlier Inside Airbnb snapshots of the same cities, tail weights, LightGBM as a third model, a three-point hyperparameter grid, a weighted blend, Duan smearing, and isotonic calibration did not clear the 1% bar. The earlier snapshots were the closest. They cut development MAE by about 0.8%, helped Seattle by about $2, and did not help Austin. Photos were not used. The 80% intervals covered 81.4% of Austin's new holdout, 80.1% of San Diego's, and 78.9% of Seattle's. The band was not widened after that look.

## Data

| Source | License | Used for |
| --- | --- | --- |
| Inside Airbnb listings | CC BY 4.0 | The rows, the price, size, room type, amenities, reviews, host fields |
| Earlier Inside Airbnb snapshots, before each study date | CC BY 4.0 | Measured as extra training rows, de-duplicated by listing, and not kept |
| Census ACS 2024 5-year and TIGER/Line tracts | Public domain | Tract income, rent, home value, density |
| OpenStreetMap via Overpass | ODbL, © OpenStreetMap contributors | Measured in v2, not in the scored model |
| potion-base-8M static embedding | MIT, MinishLab | Measured on the ad text, eight components, and not kept |

A one-shot calendar file was not a training feature. Unavailable on one scrape mixes bookings and host blocks. The daily panel on `calendar-data` is the attempt to separate them, and it is not in the model yet. Redfin, Zillow, and AirDNA were not used. Host photos were not downloaded. Snapshots dated after the study file were not used. A later price is not training data for an earlier label.

## Reproduce

From the repository root, with the pinned `requirements.txt`:

```bash
pip install -r requirements.txt
python -m evaluation.run_eval
python -m evaluation.run_occupancy
pytest
```

The first command rebuilds features, fits the development splits, writes `reports/metrics.json`, and scores the seed-23 price holdout once. It then scores seed 7 and seed 11 once. It does not fit the nights model. The second command writes the seed-31 nights holdout if it is missing, fits the nights model, and scores that holdout once. Cached city extracts, including the earlier snapshots, are already in the repo.

`evaluation/protocol.py` is the price design: the three seeds, the 1 km buffer, the host rule, and the bars for a new feature, a new blend, and a new interval. `evaluation/occupancy_protocol.py` is the nights design: seed 31, the 10% bar, and the grade cuts. Change those and you are running a different study. The held-out listing ids are in `evaluation/locked_test_ids.json`, `evaluation/confirm_ids.json`, `evaluation/final_ids.json`, and `evaluation/occupancy_ids.json`.

## Run the local app

After the artifacts exist (they are committed, or they appear when the command above finishes):

```bash
streamlit run app/streamlit_app.py
```

The app binds to `127.0.0.1:8501` only. Pick a city, search an address or click the map, enter bedrooms, bathrooms, guests, and room type. It returns a nightly rate, an 80% interval, plain-language reasons, and nearby advertised rates. It also returns predicted booked nights, estimated annual revenue (rate times nights), and an A–F grade of that revenue inside the city. A purchase price is optional and only produces a rough gross yield. A listing description, amenities, and reviews are optional. Left blank, they stay missing, which is the right input for a home that has never been listed. Reviews are not an input to the nights model. A second page shows the holdout metrics and the figures.

![CapScout local estimate](docs/app-estimate.png)

The rate is an advertised-rate estimate. It is not a quote.

## Limitations

- The label is the advertised rate, not what a guest paid.
- The point model did not improve on v3. Seed 23's lower MAE is a different, smaller split.
- Errors of tens of dollars, and intervals a hundred dollars wide, are the honest precision.
- Austin's 80% interval covered 74.9% of seed 7, and 81.4% of seed 23.
- Seattle's new holdout covered 78.9%. San Diego's new holdout is 391 listings. Seattle's is 266.
- Expensive listings are still under-predicted. The tail fixes that were tried made average error worse.
- Three cities, one protocol. This is not a national model.
- The nights figure is a model of a review-based estimate. It missed a pre-set bar against the neighbourhood median. R² is 0.073, 0.098, and 0.078.
- The grade is a within-city rank of predicted gross revenue. It is not a profit, and it matched the review-based revenue grade on about a quarter to a third of the holdout.
- A purchase price in the app is a rough gross yield only. It is not a mortgage, a tax bill, or a statement that a short-term rental is legal there.
