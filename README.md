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

**For-sale listings** come from three sources, checked in priority order:
1. **Redfin CSV** (free, no API key) — download a CSV from redfin.com and upload it in Settings
2. **RentCast API** — for-sale listings with an API key (50 free calls/month)
3. **Realtor.com via RapidAPI** — for-sale listings with property photos

If no for-sale data is available, the dashboard falls back to showing Airbnb listings ranked by predicted performance.

**Street View photos** are fetched from the Google Maps Static API for properties scoring above 30. Images are cached permanently to disk so each location is only fetched once. A monthly counter enforces the free-tier limit (~25,000 requests/month). The API key stays server-side — the frontend loads images through a proxy route.

## Machine Learning Pipeline

CapScout trains **two models per city** from the Inside Airbnb dataset:

- **Price Model** — predicts the optimal nightly rate from property features (bedrooms, bathrooms, location, amenities, property type)
- **Occupancy Model** — predicts monthly occupancy with seasonal awareness (uses calendar data to capture month-by-month booking patterns)

Training compares 5 regression algorithms (Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM) using randomized hyperparameter search with 5-fold cross-validation, then selects the best performer automatically. Models are trained independently per city so predictions reflect local market conditions.

When you view a property, the models predict its nightly rate and occupancy, then calculate monthly/annual revenue projections. These feed into the financial model that computes the CapScout Score.

> **Note:** The models are trained on Airbnb listing data which includes review scores, host metrics, and booking history — features that don't exist for homes listed for sale. When scoring for-sale properties, these features are filled with training-set medians, meaning they contribute no discriminating power. A future improvement would be to retrain with only universally-available features (bedrooms, bathrooms, location, amenities, property type) to reduce noise for for-sale predictions.

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

# Train ML models (~5-10 min per city)
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
  cities.py                  # City registry + seasonal factors
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
    feature_engineering.py   # Feature extraction from listings
    train_model.py           # Model training pipeline
  scripts/
    download_data.py         # Inside Airbnb data downloader
  data/
    cities/                  # Per-city Airbnb data (gitignored)
    streetview_cache/         # Cached Street View images (gitignored)

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

**Backend:** Python, Flask, scikit-learn, XGBoost, LightGBM, pandas, NumPy, joblib

**Frontend:** React 19, Vite, MapLibre GL, Recharts, Axios, React Router
