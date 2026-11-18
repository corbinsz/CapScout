# CapScout

Live demo: [capscout.vercel.app](https://capscout.vercel.app)

CapScout is a small full-stack app that looks at homes for sale in 14 U.S. markets and asks a narrow question: if you ran the place as a short-term rental, how would the estimated income compare with the mortgage, taxes, insurance, and operating costs? Each listing gets a CapScout score from 0 to 100. The score is a worksheet on top of two city-level XGBoost models (nightly rate and occupancy). It is a screening tool, not an appraisal.

The frontend is React and Vite. The API is Flask. Models and listing files for all 14 cities, including Hawaii, are in the repo.

## How it works

**Data.** Training rows come from [Inside Airbnb](https://insideairbnb.com) listing files. The cleaner keeps entire homes and drops nightly prices outside the 2nd–98th percentile. Each listing is expanded to 12 months. The month's price is the listing's asking price times a hand-set seasonal factor for that city. The month's occupancy is `1 - availability_30 / 30`, clipped, times the same kind of seasonal factor. `availability_30` counts nights the host has open over the next month, so it mixes unbooked nights and nights the host blocked. The current training path does not read the Inside Airbnb calendar file.

For-sale homes come from a Redfin GIS download (`scripts/download_redfin.py`), with vacant land, commercial, and large multifamily filtered out. Redfin rows have no nightly-rate label, so they are inputs at prediction time, not a test set.

Austin, Dallas, and Seattle have an assessor-join path (sqft, year built, lot size). Those enrichment CSVs are not in the repo, and the saved models don't store a match rate, so this README doesn't quote one. Other cities leave those fields empty; at train and predict time they are filled with the column median.

**Models.** Each city has its own price model and occupancy model. Before fitting, listings are clustered with k-means on coordinates (up to about 20 zones). The training target for a row is the median price or occupancy of listings in the same zone, bedroom count, and month, as long as the group has at least 3 listings. The model is learning "what does a typical home of this size in this area go for this month," which is a different question from "what does this host charge."

Features are limited to things Redfin also has: bedrooms, bathrooms, an estimated guest count (`bedrooms * 2`), latitude, longitude, distance to the city center, five property-type flags, sqft, year built, lot size, and three ratios derived from those, plus `sin_month` and `cos_month`. Review scores, amenities, and host stats are left out because a for-sale listing would just get a constant fill.

The shipped models were tuned with `RandomizedSearchCV` (50 draws, 5-fold CV on negative MSE) and then refit on every row. There is no holdout file next to the joblibs. `python -m models.train_model` prints two numbers. Train R² is `sklearn.metrics.r2_score` of that refit model on the rows it just saw. The "CV R²" line is `1 - (mean fold MSE) / variance(all smoothed y)`. The variance is the full sample, not each held-out fold, and the medians are computed before the split. That print is what an older README table reported as price CV R² of about 0.89–0.995. An earlier intro line said R² 0.70–0.86. Nothing in the repo computes that range.

**Scoring.** For a Redfin home the API predicts all 12 months, clips occupancy to 5–95%, and sums `nightly rate × occupancy × days in the month`. That annual figure goes into `score_property`:

| Piece | Points | Curve in the code |
| --- | ---: | --- |
| Gross yield (revenue / price) | 25 | 20% yield fills the bar |
| Cash-on-cash (cash flow / down payment) | 25 | 0% scores 5 points; 20% scores 25 |
| Cap rate (NOI / price) | 20 | 10% cap rate fills the bar |
| DSCR (NOI / debt service) | 20 | 1.0 is the middle of the bar; 1.5 fills it |
| Predicted occupancy | 10 | 75% occupancy fills the bar |

Default financing is 20% down, 7% interest, 30 years. Property tax, insurance, and the annual permit fee come from `backend/cities.py`. Operating costs are itemized: 3% platform fee, city cleaning cost per turnover, $25 supplies per turnover, utilities that move partly with occupancy, $125 a month for internet and misc, and 1% of price for maintenance. Filters on the dashboard can override the loan terms.

On the cards, 70+ is Excellent (purple), 55–69 Good (green), 45–54 Average (yellow). Below 45 the badge is red; the label splits that into Below average (30–44) and Poor (under 30).

Confidence is a weighted count of training listings within 1, 3, and 5 km. The dashboard hides "low" confidence by default. Street View photos are requested for scores of 30 and up, through the API, with a 25,000 request counter for the month.

## Holdout check

The numbers below are from `python -m scripts.eval_holdout`, run on the listing files in this repo. It does not score the saved models (those have already seen every listing). It holds out 20% of listing IDs with seed 42, fits area medians on the training listings only, and fits the fixed XGBoost in `_quick_train` (300 trees, depth 8, learning rate 0.08). Metrics are `sklearn.metrics.r2_score` and mean absolute error on the held-out listing-months.

"Vs median" is the model against the area-median target it was trained on. "Vs listing" is the same predictions against that listing's own seasonally scaled price or occupancy. MAE is in dollars per night.

| City | Listings | Price R² vs median | Price R² vs listing | Price MAE | Occ. R² vs listing |
| --- | ---: | ---: | ---: | ---: | ---: |
| Austin | 8,722 | 0.977 | 0.440 | 76.3 | 0.034 |
| Nashville | 6,014 | 0.969 | 0.423 | 50.8 | 0.065 |
| San Diego | 9,628 | 0.992 | 0.531 | 85.3 | 0.041 |
| Dallas | 4,432 | 0.966 | 0.514 | 57.5 | -0.070 |
| Asheville | 2,651 | 0.971 | 0.097 | 70.3 | 0.004 |
| Seattle | 4,793 | 0.966 | 0.464 | 38.5 | 0.010 |
| Denver | 3,520 | 0.946 | 0.468 | 45.2 | -0.101 |
| New Orleans | 5,201 | 0.969 | 0.380 | 60.4 | -0.083 |
| Washington, D.C. | 3,552 | 0.943 | 0.455 | 43.5 | -0.038 |
| Boston | 2,389 | 0.871 | 0.511 | 76.6 | -0.015 |
| Las Vegas | 8,522 | 0.982 | 0.297 | 75.8 | 0.014 |
| Fort Lauderdale | 9,493 | 0.987 | 0.485 | 55.6 | -0.037 |
| Fort Worth | 1,390 | 0.905 | 0.395 | 37.4 | -0.057 |
| Hawaii | 29,082 | 0.987 | 0.342 | 161.4 | 0.008 |

Price R² against the median target sits between 0.87 and 0.99. The target is a median of location, bedrooms, and month, and those are features, so a high number mostly means the model can reproduce that median. Price R² against the listing's own rate is the column to quote for a specific home: 0.10 in Asheville, 0.53 in San Diego, and about 0.30 to 0.51 in the other cities. Absolute error runs from about $37 a night in Fort Worth to about $85 in San Diego, and $161 in Hawaii.

Occupancy R² against the median target is 0.64–0.99 (Fort Worth is the low end). Against the listing's own availability-based rate it is about −0.10 to 0.07, with an absolute error of about 0.20–0.27. The occupancy model is tracking the area median of a noisy proxy.

The same script also fits a model to the listing price itself, same features and same split. That price R² is about 0.48–0.67. Occupancy stays weak (about −0.10 to 0.22). The features carry a moderate signal for asking price. They carry very little for this occupancy proxy, and training on the median gives some of the price signal away, especially in Hawaii and Asheville.

## Cities

Tax and permit figures are the assumptions in `backend/cities.py`, which the scorer actually uses. They are not a citation of current local law. Every city below has a price model, an occupancy model, an Inside Airbnb listings file, and a Redfin file in the repo.

| City | Property tax | Annual permit in the model |
| --- | ---: | ---: |
| Austin, TX | 1.80% | $0 |
| Nashville, TN | 1.15% | $350 |
| San Diego, CA | 0.76% | $1,000 |
| Dallas, TX | 1.80% | $0 |
| Asheville, NC | 0.63% | $200 |
| Seattle, WA | 0.90% | $75 |
| Denver, CO | 0.50% | $200 |
| New Orleans, LA | 1.10% | $500 |
| Washington, D.C. | 0.85% | $100 |
| Boston, MA | 1.06% | $200 |
| Las Vegas, NV | 0.60% | $1,000 |
| Fort Lauderdale, FL | 1.10% | $250 |
| Fort Worth, TX | 1.80% | $0 |
| Hawaii | 0.30% | $500 |

Hawaii is one statewide model. The map centers on Honolulu. Insurance assumptions in the same file run from 0.3% of price (Seattle, Las Vegas) to 0.8% (New Orleans, Fort Lauderdale).

## Stack

Python, Flask, pandas, scikit-learn, XGBoost, gunicorn on the API. React 19, Vite, MapLibre, Recharts on the client. The UI in this repo is deployed on Vercel. Locally, Vite proxies `/api` to the Flask app.

## Run it locally

Python 3.10+ and Node 18+.

```bash
cd backend
pip install -r requirements.txt
python app.py
```

```bash
npm install
npm run dev
```

The site is at [http://localhost:3000](http://localhost:3000). The API is at [http://localhost:5000](http://localhost:5000). A `GOOGLE_MAPS_API_KEY` in `backend/.env` turns on Street View. Without it, the rest of the app still runs.

```bash
cd backend
python -m pytest                  # score math; pip install pytest first
python -m scripts.eval_holdout   # the table above, about a minute
```

Retraining (`python -m models.train_model austin`, or `--all`) overwrites the joblib files. The download scripts are `python -m scripts.download_data` and `python -m scripts.download_redfin`.

## Limitations

- The saved models were fit on every training listing. The holdout table is a separate refit with fixed hyperparameters, so it is a fair check of the pipeline, not a score of the exact file the API loads.
- Nightly rates are asking prices, seasonally scaled. Occupancy is a 30-day availability proxy times a seasonal curve written in `cities.py`. Neither is booked revenue.
- Area-median targets make the fit look tight and leave a lot of host-level price variation on the table. Photos, reviews, and amenities are not features.
- Sqft and year built are only as good as the assessor join, and that file is not shipped here.
- Loan terms, tax rates, and permit fees are editable assumptions. A high score means "these inputs cash-flow under these assumptions."
- Hawaii's error is large in dollars because prices there spread widely inside a single statewide model.
