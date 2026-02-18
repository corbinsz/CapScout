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

**Street View photos** are fetched from the Google Maps Static API for properties scoring above 30. Images are cached permanently to disk so each location is only fetched once. A monthly counter enforces the free-tier limit (~25,000 requests/month). The API key stays server-side — the frontend loads images through a proxy route.

## Machine Learning Pipeline

CapScout trains **two models per city** from the Inside Airbnb dataset:

- **Price Model** — predicts the expected nightly rate from property features and location
- **Occupancy Model** — predicts monthly occupancy with seasonal awareness (uses calendar data to capture month-by-month booking patterns)

### Prediction Strategy: Area Median Targeting

The core challenge in STR revenue prediction is that individual Airbnb hosts set wildly different prices for similar properties — two identical 3-bedroom homes on the same street can list at $150/night and $400/night depending on the host's pricing strategy, photo quality, and marketing. Trying to predict individual host pricing from property features alone yields ~54% accuracy (R²=0.54).

CapScout solves this with **area median targeting**: instead of training the model to predict what each individual listing charges, we train it to predict what a **typical listing in that area earns**. The training pipeline:

1. **Clusters** all Airbnb listings into ~20 geographic zones using k-means on coordinates
2. **Groups** listings by (zone, bedroom count, month) — e.g., "3-bedrooms in East Austin in July"
3. **Replaces** each listing's individual price/occupancy with its group's **median** as the training target
4. The model learns: *"a typical 3BR in this neighborhood earns $X/night in July"*

This eliminates host-specific noise from the training signal. For occupancy, this also mitigates the problem of hosts blocking calendar dates (which looks identical to zero bookings in the data) — area medians smooth out individual host availability patterns.

**Why area medians and not raw values?** Individual Airbnb occupancy is extremely noisy. In the Austin training data, 57% of listing-months show 0-20% occupancy — mostly because hosts block dates, not because demand is zero. Area medians give a more honest signal of "what can you expect in this location" which is the right question for evaluating a for-sale property.

For investors, this answers: *"What would a typical STR at this location earn?"* — not *"What does one specific Airbnb host happen to charge?"*

### Feature Selection

Only features that exist for **both** Airbnb training data and for-sale properties are included (40 total). Airbnb-only features like review scores, host metrics, and booking activity were removed because they get filled with meaningless constants for for-sale properties. The model uses:

- **Property basics** — bedrooms, bathrooms, accommodates, beds
- **Location** — latitude, longitude, distance to city center
- **Amenities** — 18 binary flags (pool, hot tub, kitchen, parking, A/C, etc.)
- **Property type** — 8 binary flags (home, condo, townhouse, cabin, etc.)
- **Physical features** — sqft, year built, lot size, sqft/bedroom, property age, lot acres (from county assessor enrichment; NaN-filled for unenriched cities)
- **Seasonality** — cyclical month encoding (sin/cos) for month-aware predictions

Physical features are the key bridge between training data and for-sale properties: Redfin listings already have sqft, year_built, and lot_size, and now the training data does too (via assessor enrichment). For enriched cities, the model uses real structural data instead of training medians.

### Model Selection

Training compares **LightGBM and XGBoost** — the two best algorithms for tabular regression. Both handle missing values natively (important for optional features like sqft) and are histogram-based for fast training. The winner is selected automatically by lowest cross-validated MSE (50 iterations of randomized hyperparameter search, 5-fold CV). Models are trained independently per city.

The train/test split is done **by listing ID** (not by row) so all months of a listing stay together. This prevents data leakage and ensures honest out-of-sample evaluation — the test set contains only listings the model has never seen. Target smoothing (area medians) is computed from training data only — test listings are smoothed using training-derived cluster medians, never their own neighborhoods.

### Out-of-Sample Accuracy

All metrics below are on held-out test sets (20% of listings, never seen during training). Train R² shows model memorization; the gap between train and test R² indicates overfitting.

**Price Models (nightly rate prediction):**

| City | Train R² | Test R² | Gap | Avg Error | MAPE |
|------|:--------:|:-------:|:---:|:---------:|:----:|
| Nashville | 0.999 | **0.992** | 0.01 | $17/night | 7.4% |
| Dallas | 1.000 | **0.971** | 0.03 | $5/night | 2.7% |
| Austin | 0.999 | **0.967** | 0.03 | $9/night | 5.2% |
| Seattle | 0.999 | **0.959** | 0.04 | $8/night | 3.6% |
| Asheville | 1.000 | **0.949** | 0.05 | $2/night | 1.3% |
| San Diego | 0.998 | **0.751** | 0.25 | $20/night | 5.3% |

**Occupancy Models (monthly occupancy rate):**

| City | Train R² | Test R² | Gap | Avg Error |
|------|:--------:|:-------:|:---:|:---------:|
| Asheville | 0.998 | **0.969** | 0.03 | 1.5 pts |
| Seattle | 1.000 | **0.917** | 0.08 | 4.5 pts |
| Nashville | 1.000 | **0.915** | 0.08 | 2.8 pts |
| San Diego | 1.000 | **0.890** | 0.11 | 3.9 pts |
| Austin | 0.999 | **0.888** | 0.11 | 3.5 pts |
| Dallas | 0.995 | **0.785** | 0.21 | 1.8 pts |

**Notes on accuracy:**
- Train R² > 0.99 for all models is normal for gradient boosted trees — they can memorize training data. The test R² is what matters.
- All 6 cities now have "GOOD" ratings for both price and occupancy models.
- Austin and Seattle benefit from county assessor enrichment (real sqft/year_built for 85-88% of training listings).
- San Diego's price model has a wider gap (R² 0.75) due to extreme price variance between beachfront luxury and inland properties. Calendar price outliers are capped at P99 to mitigate this.
- Dallas uses synthetic monthly data (no Inside Airbnb calendar available), so its occupancy model has less real booking signal but still achieves 79% accuracy.
- Nashville trains on 2025 listings matched with 2024 calendar data (the most recent snapshot with both).
- Predictions are not post-processed or inflated — the model's raw output is clipped to [5%, 95%] and returned as-is.

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

# Evaluate model accuracy (shows train vs test R², overfitting gaps)
python -m scripts.eval_models

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
    train_model.py           # Model training pipeline (LightGBM + XGBoost)
  scripts/
    download_data.py         # Inside Airbnb data downloader
    enrich_listings.py       # County assessor enrichment pipeline
    eval_models.py           # Model evaluation (train vs test R², overfitting check)
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

**Backend:** Python, Flask, scikit-learn, XGBoost, LightGBM, pandas, NumPy, SciPy, joblib

**Frontend:** React 19, Vite, MapLibre GL, Recharts, Axios, React Router
