# How CapScout works

CapScout has two parts.

The website is a React app. Vercel builds it with `npm run build` and serves the `dist` folder. Opening a link such as `/deals` still returns `index.html`, and the app picks the page. `/api/...` on Vercel is forwarded to the Python API on Render so a preview can call the backend without a separate address.

The API is a Flask app in `backend/`. Render starts it with gunicorn. It scores homes, stores accounts, and talks to listing providers. Models live in `backend/models/trained/<city>/`.

## A request for deals

1. The deals page asks `GET /api/properties?city=austin`.
2. The API checks the city name against the list in `backend/cities.py`. A name with `..` in it is rejected, so a request cannot write outside the data folders.
3. The listing provider is chosen in `backend/services/providers/registry.py`.
   - If `RENTCAST_API_KEY` is set, listings come from RentCast and are cached on disk.
   - Else if `ALLOW_REDFIN_FALLBACK=1`, a CSV already saved for that city is read. The app does not call Redfin.
   - Else the page says no licensed feed is configured and shows Inside Airbnb rows so the screen is not blank. Those rows are active rentals, not homes for sale.
4. Every home that passes the price and bedroom filters is scored in one model call, then sorted. The old code scored only the 100 cheapest homes and called that a ranking. That is fixed.
5. The score uses the mortgage, taxes, insurance, permit, and operating costs for that city. The sentence on the card is produced by `explain_deal` in `backend/services/scoring_service.py`.

## The model

Each city has two XGBoost models: nightly rate and occupancy. Price is trained on the log of the rate and turned back into dollars when it is loaded (`backend/models/estimators.py`). Seasonality is applied when predicting, from the factors in `cities.py`. The model file itself is the annual pattern, not twelve copies of the same home.

Training data is Inside Airbnb, cleaned in `backend/models/feature_engineering.py`. Entire homes only. Extreme prices are cut. Occupancy is `1 - availability_30 / 30`. A host who blocks a date looks like a booking, so occupancy error stays large. That limit is stated in the product.

Evaluation is in `backend/models/benchmark.py` and `docs/ml_benchmark.json`. The selection score is mostly raw nightly-rate error on a 20% holdout of whole listings, plus a time split using `last_review`. An area-median model won an earlier score that rewarded predicting the area median. It was worse on real prices, so it was not kept.

## Accounts and money

Accounts are rows in a SQLite file on the server (`backend/data/accounts.sqlite`, not in git). Passwords are hashed. The session is an HttpOnly cookie. Sign-in from the browser needs `SECRET_KEY` in production so the cookie still works after a restart.

The Top deals CSV is `GET /api/reports/top-deals`. It returns 401 if you are signed out and 402 if you are not Pro. Stripe checkout is `POST /api/billing/checkout`. With no live key it returns a stub and charges nothing. A webhook marks a user Pro only when Stripe’s signature checks out.

Weekly alerts are `POST /api/alerts`. They are stored. No email is sent.

## Data licenses

Inside Airbnb’s public files are under Creative Commons Attribution 4.0. You may use them commercially if you credit Inside Airbnb, link the license, and say you changed the data. Their project also asks you not to republish the raw files and not to scrape their site. CapScout trains on a local copy and does not rehost the raw dump. Credit them when you publish market stats.

RentCast is a paid, licensed API. The key stays in the server environment.

Redfin’s unofficial GIS endpoint is not a license for a commercial product. `backend/scripts/download_redfin.py` exits unless `ALLOW_REDFIN_DOWNLOAD=1`.

County assessor files, where present, are public assessment records used to add square footage to training rows.

## Privacy defaults

Analytics and client-error logging are off until `ANALYTICS_ENABLED` and `CLIENT_ERROR_LOG` are set, and the website only calls them when `VITE_ANALYTICS=1` or `VITE_ERROR_REPORTING=1` at build time. Counts are by event name and day. They are not tied to an email. Error lines that contain `@` are dropped.
