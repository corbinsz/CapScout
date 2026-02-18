# CapScout - Short-Term Rental Investment Analyzer

CapScout is a full-stack web application that helps real estate investors evaluate properties for short-term rental (STR) potential across multiple U.S. cities. It combines real listing data, machine learning predictions, and financial modeling to produce a single **CapScout Score (0-100)** for every property.

## Supported Cities

| City | Property Tax | STR Permit | Notes |
|------|-------------|------------|-------|
| Austin, TX | 1.8% | Free | SXSW + ACL/F1 peaks |
| Nashville, TN | 1.15% | $350/yr | Summer tourism + CMA Fest |
| San Diego, CA | 0.76% | $1,000/yr | Beach city summer peak |
| Dallas, TX | 1.8% | Free | Business travel market |
| Asheville, NC | 0.63% | $200/yr | Fall foliage peak |
| Seattle, WA | 0.9% | $75/yr | Pacific NW summer peak |

Switch between cities using the dropdown in the navbar. Each city has its own trained ML models, property tax rates, seasonal factors, and market data.

## Data Sources

**Airbnb market data** comes from [Inside Airbnb](https://insideairbnb.com), a public dataset providing real listing details (nightly rates, occupancy, amenities, property type, location) and calendar availability for every active Airbnb in a city. This powers the ML models and analytics dashboard.

**County assessor data** enriches the training data with physical property attributes (sqft, year built, lot size) from public tax assessor records. Assessor records are spatially matched to Airbnb listings using KD-tree nearest-neighbor search within a ~200m radius, disambiguated by bedroom count. This is available for cities with free bulk data downloads:

| City | Source | Match Rate | Features Added |
|------|--------|:----------:|----------------|
| Seattle | King County Assessor | 85% | sqft, year_built, lot_sqft, bedrooms |
| Austin | Travis CAD | 88% | sqft, year_built, bedrooms, bathrooms |
| Others | Not available | — | Features filled with training medians |

The enrichment pipeline (`scripts/enrich_listings.py`) downloads assessor CSVs, queries county ArcGIS services for parcel coordinates, performs spatial matching, and outputs an `assessor_enrichment.csv` per city that the training pipeline automatically picks up.

**For-sale listings** come from three sources, checked in priority order:
1. **Redfin CSV** (free, no API key) — download a CSV from redfin.com and upload it in Settings
2. **RentCast API** — for-sale listings with an API key (50 free calls/month)
3. **Realtor.com via RapidAPI** — for-sale listings with property photos

If no for-sale data is available, the dashboard falls back to showing Airbnb listings ranked by predicted performance.

**Redfin CSV filtering pipeline:** The raw CSV goes through two filter stages before reaching the dashboard:

1. **Import filter** — removes vacant land, commercial properties, parking, multi-family 5+ units, construction plans, listings with no address/price, and 0-bedroom units. This count is shown on the Settings page.
2. **Dashboard filter** — removes listings where the price parses to $0 (pending, withdrawn, or "coming soon" listings that Redfin includes without a real price).

For example, a 4,290-row Austin CSV might show 3,735 on the Settings page and 2,572 on the dashboard.

**Street View photos** are fetched from the Google Maps Static API for properties scoring above 30. Images are cached permanently to disk so each location is only fetched once. A monthly counter enforces the free-tier limit (~25,000 requests/month). The API key stays server-side — the frontend loads images through a proxy route.

## Machine Learning Pipeline

CapScout trains **two XGBoost models per city** from the Inside Airbnb dataset:

- **Price Model** — predicts the expected nightly rate from property features and location
- **Occupancy Model** — predicts monthly occupancy with seasonal awareness (uses calendar data to capture month-by-month booking patterns)

### The Complete Pipeline

```
Inside Airbnb data (listings + calendar)
    --> Clean & filter (entire homes only, P2-P98 listing prices, P95 calendar prices)
    --> K-means geographic clustering (~20 zones)
    --> Target smoothing (group medians by zone x bedrooms x month)
    --> XGBoost + RandomizedSearchCV (50 iters, 5-fold CV)
    --> Final model retrained on 100% of data
    --> Predict on Redfin for-sale properties
```

**Data cleaning notes:** Calendar prices are capped at P95 (95th percentile) rather than P99 because some cities (notably Nashville) have tens of thousands of entries at $9,999 — hosts use placeholder pricing to block dates. P95 removes this garbage while keeping legitimately expensive listings.

**No hold-out split.** The model trains on 100% of Airbnb data. Cross-validation folds handle evaluation during hyperparameter search, and the final model is retrained on all data with the best parameters. The real "test set" is Redfin for-sale properties — data the model has never seen and that comes from a completely different distribution.

### Prediction Strategy: Area Median Targeting

The core challenge in STR revenue prediction is that individual Airbnb hosts set wildly different prices for similar properties — two identical 3-bedroom homes on the same street can list at $150/night and $400/night depending on the host's pricing strategy, photo quality, and marketing. Trying to predict individual host pricing from property features alone yields ~54% accuracy.

CapScout solves this with **area median targeting**: instead of training the model to predict what each individual listing charges, we train it to predict what a **typical listing in that area earns**. The training pipeline:

1. **Clusters** all Airbnb listings into ~20 geographic zones using k-means on coordinates
2. **Groups** listings by (zone, bedroom count, month) — e.g., "3-bedrooms in East Austin in July"
3. **Replaces** each listing's individual price/occupancy with its group's **median** as the training target
4. The model learns: *"a typical 3BR in this neighborhood earns $X/night in July"*

This eliminates host-specific noise from the training signal. For occupancy, this also mitigates the problem of hosts blocking calendar dates (which looks identical to zero bookings in the data) — area medians smooth out individual host availability patterns.

**Why area medians and not raw values?** Individual Airbnb occupancy is extremely noisy. In the Austin training data, 57% of listing-months show 0-20% occupancy — mostly because hosts block dates, not because demand is zero. Area medians give a more honest signal of "what can you expect in this location" which is the right question for evaluating a for-sale property.

For investors, this answers: *"What would a typical STR at this location earn?"* — not *"What does one specific Airbnb host happen to charge?"*

### Feature Selection (17 features + 2 seasonal)

Only features that exist for **both** Airbnb training data and Redfin for-sale properties are included. Amenities (Redfin has no amenity data), review scores, host metrics, and booking activity were all removed — they get median-filled as constants for for-sale properties, contributing noise without signal.

| Feature | Redfin Source | Type |
|---------|--------------|------|
| `bedrooms` | BEDS | real |
| `bathrooms_clean` | BATHS | real |
| `accommodates` | bedrooms x 2 | derived |
| `latitude` | LATITUDE | real |
| `longitude` | LONGITUDE | real |
| `distance_to_center` | haversine(lat/lng, city center) | derived |
| `is_home` | PROPERTY TYPE | flag |
| `is_condo` | PROPERTY TYPE | flag |
| `is_townhouse` | PROPERTY TYPE | flag |
| `is_cabin` | PROPERTY TYPE | flag |
| `is_cottage` | PROPERTY TYPE | flag |
| `sqft` | SQUARE FEET | real |
| `year_built` | YEAR BUILT | real |
| `lot_sqft` | LOT SIZE | real |
| `sqft_per_bedroom` | sqft / bedrooms | derived |
| `property_age` | 2025 - year_built | derived |
| `lot_acres` | lot_sqft / 43560 | derived |
| `sin_month` | month (1-12) | seasonal |
| `cos_month` | month (1-12) | seasonal |

Physical features (sqft, year_built, lot_sqft) are the key bridge between training data and for-sale properties: Redfin listings already have these, and now the training data does too (via county assessor enrichment). For enriched cities, the model uses real structural data instead of training medians.

### XGBoost + Hyperparameter Tuning

**XGBoost** is the sole model — appropriate for our data sizes (40K-156K rows per city), handles missing values natively (important for optional features like sqft in unenriched cities), and is histogram-based for fast training.

Hyperparameters are tuned via **RandomizedSearchCV** (50 random combinations, 5-fold cross-validation):

| Parameter | Range | Controls |
|-----------|-------|----------|
| `n_estimators` | 200-800 | Number of trees |
| `max_depth` | 4-12 | Tree depth |
| `learning_rate` | 0.01-0.20 | Step size |
| `subsample` | 0.7-1.0 | Row sampling per tree |
| `colsample_bytree` | 0.5-1.0 | Feature sampling per tree |
| `reg_alpha` | 0-1 | L1 regularization |
| `reg_lambda` | 0.5-2.0 | L2 regularization |
| `min_child_weight` | 1-10 | Min leaf weight |

### Model Accuracy

Training reports two R² metrics per target:
- **Train R²** — predict on all training data with the final refitted model. Measures how well the model fits the data.
- **CV R²** — average R² across the 5 held-out CV folds. Measures generalization to unseen data.

A small gap between Train R² and CV R² means the model generalizes well. A large gap indicates overfitting.

| City | Price CV R² | Occ CV R² | Notes |
|------|:----------:|:---------:|-------|
| San Diego | 0.996 | 0.971 | Best overall |
| Dallas | 0.976 | 0.831 | Synthetic calendar |
| Austin | 0.971 | 0.936 | Assessor enriched |
| Asheville | 0.965 | 0.967 | Small but clean data |
| Seattle | 0.951 | 0.920 | Assessor enriched |
| Nashville | 0.921 | 0.933 | Fixed with P95 cap |

### How Predictions Work

When you view a for-sale property, CapScout:

1. Extracts property features (bedrooms, bathrooms, sqft, year_built, lot_size, property type, location)
2. For enriched cities, uses actual sqft/year_built/lot_size from Redfin; for others, fills with training medians
3. Predicts nightly rate and occupancy for each of the 12 months (seasonal via sin/cos encoding)
4. Computes annual revenue from the monthly predictions
5. Runs the financial model (mortgage, taxes, insurance, expenses) to produce the CapScout Score

### Prediction Confidence

Confidence is based on geographic density of training data near the property:
- **High** — 50+ Airbnb listings within 3km (strong training signal)
- **Medium** — 15-49 listings within 3km
- **Low** — fewer than 15 listings (predictions less reliable)

By default, the dashboard hides low-confidence properties. This can be toggled with the "Confident Predictions Only" filter.

## The CapScout Score

The score (0-100) is a weighted composite of five financial metrics:

| Component | Max Points | What It Measures |
|-----------|-----------|------------------|
| Gross Yield | 25 | Annual revenue / purchase price |
| Cash-on-Cash ROI | 25 | Annual cash flow / down payment |
| Cap Rate | 20 | Net operating income / purchase price |
| DSCR | 20 | Debt service coverage ratio (NOI / debt service) |
| Occupancy | 10 | Predicted booking rate |

**Score interpretation:** 75+ Excellent | 55-74 Good | 40-54 Fair | <40 Poor

Financial modeling assumes 20% down payment, 7% mortgage rate, 30-year term, and city-specific property tax / insurance / expense rates.

## Key Features

- **Dashboard** — paginated, filterable property list with scores, revenue predictions, and occupancy estimates
- **Interactive Map** — MapLibre GL map centered on the selected city with color-coded score markers
- **Analytics** — market overview with neighborhood rankings, price distributions, bedroom breakdowns, and revenue charts
- **Property Detail** — deep dive with ML predictions, financial analysis, comparable listings, Street View photo, and a mini map
- **Prediction Process** — explains the full pipeline from data collection through scoring, step by step
- **Settings** — upload Redfin CSVs, configure API keys, and test the prediction model with custom inputs

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+

### Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys (all optional)
cat > .env << 'EOF'
RENTCAST_API_KEY=your_key_here
RAPIDAPI_KEY=your_key_here
GOOGLE_MAPS_API_KEY=your_key_here
EOF

# Download Inside Airbnb data for all cities
python -m scripts.download_data

# (Optional) Enrich training data with county assessor records
# Currently supports Seattle (King County) and Austin (Travis CAD)
python -m scripts.enrich_listings

# Train ML models (~20 min per city with enrichment, ~10 min without)
python -m models.train_model --all

# Start Flask server
python app.py
```

### Frontend Setup

```bash
# From project root
npm install
npm run dev
```

The frontend runs on `http://localhost:5173` and the backend API on `http://localhost:5000`.

### Environment Variables

| Variable | Source | Purpose |
|----------|--------|---------|
| `RENTCAST_API_KEY` | [RentCast](https://www.rentcast.io/) | For-sale property listings |
| `RAPIDAPI_KEY` | [RapidAPI](https://rapidapi.com/) | Realtor.com listings |
| `GOOGLE_MAPS_API_KEY` | [Google Cloud Console](https://console.cloud.google.com/) | Street View property photos |

All keys are optional. Without any keys, the app works with Redfin CSV uploads and Airbnb data only.

## Project Structure

```
backend/
  app.py                     # Flask app factory
  config.py                  # Paths, env vars, directories
  cities.py                  # City registry + seasonal factors + assessor config
  routes/
    properties.py            # Property listing + detail endpoints
    analytics.py             # Market analytics endpoints
    predict.py               # ML prediction endpoint
    settings.py              # CSV upload + config endpoints
    streetview.py            # Street View image proxy
  services/
    data_service.py          # Airbnb + for-sale data loading
    prediction_service.py    # ML model inference
    scoring_service.py       # CapScout Score calculation
    redfin_service.py        # Redfin CSV parsing
    streetview_service.py    # Google Street View with disk cache
  models/
    feature_engineering.py   # Feature extraction, assessor merge, target smoothing
    train_model.py           # Model training pipeline (XGBoost)
  scripts/
    download_data.py         # Inside Airbnb data downloader
    enrich_listings.py       # County assessor enrichment pipeline
    download_redfin.py       # Redfin CSV download helper
  data/
    cities/                  # Per-city data (gitignored)
      {city}/
        listings.csv.gz      # Inside Airbnb listings
        calendar.csv.gz      # Inside Airbnb calendar
        assessor_raw/        # Raw assessor downloads (CSVs, centroids)
        assessor_enrichment.csv  # Matched listing→assessor records

src/
  App.jsx                    # React Router setup
  main.jsx                   # App entry point
  index.css                  # Global styles
  pages/
    Dashboard.jsx            # Property list + map view
    Analytics.jsx            # Market analytics charts
    PropertyPage.jsx         # Single property detail
    PredictionProcess.jsx    # How-it-works methodology page
    Settings.jsx             # Configuration page
  components/
    layout/                  # Navbar, CitySelector, SearchBar, Logo
    map/                     # MapView, PropertyPopup
    charts/                  # NeighborhoodChart, BedroomBreakdownChart
    filters/                 # FilterPanel
    properties/              # PropertyCard
  hooks/                     # useProperties, useAnalytics
  services/                  # api, propertyService, analyticsService
  contexts/                  # React context providers
  utils/                     # constants
```

## Tech Stack

**Backend:** Python, Flask, scikit-learn, XGBoost, pandas, NumPy, SciPy, joblib

**Frontend:** React 19, Vite, MapLibre GL, Recharts, Axios, React Router
