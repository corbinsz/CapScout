# CapScout - Short-Term Rental Investment Analyzer

CapScout is a full-stack web application that helps real estate investors evaluate for-sale properties across 14 U.S. cities. It predicts short-term rental income using machine learning, then scores each property on **how well that income covers ownership costs** (mortgage, taxes, insurance) -- not on rental income alone. The result is a single **CapScout Score (0-100)** measuring overall investment viability.

## Supported Cities (14)

| City | Property Tax | STR Permit | Notes |
|------|-------------|------------|-------|
| Austin, TX | 1.8% | Free | Assessor enriched |
| Nashville, TN | 1.15% | $350/yr | Summer + CMA Fest |
| San Diego, CA | 0.76% | $1,000/yr | Beach city summer peak |
| Dallas, TX | 1.8% | Free | Assessor enriched |
| Asheville, NC | 0.63% | $200/yr | Fall foliage peak |
| Seattle, WA | 0.9% | $75/yr | Assessor enriched |
| Denver, CO | 0.5% | $200/yr | Ski + summer tourism |
| New Orleans, LA | 1.1% | $250/yr | Mardi Gras + Jazz Fest |
| Washington, D.C. | 0.85% | $200/yr | Cherry blossom + gov travel |
| Boston, MA | 1.0% | $200/yr | Fall foliage + university |
| Las Vegas, NV | 0.6% | Free | Year-round tourism |
| Fort Lauderdale, FL | 0.9% | Free | Snowbird winter peak |
| Fort Worth, TX | 1.8% | Free | Stockyards tourism |
| Honolulu, HI | 0.35% | Free | Year-round beach demand |

Switch between cities using the dropdown in the navbar. Each city has its own trained ML models, property tax rates, seasonal factors, and market data.

## Data Sources

**Airbnb market data** comes from [Inside Airbnb](https://insideairbnb.com), a public dataset providing real listing details (nightly rates, occupancy, amenities, property type, location) and calendar availability for every active Airbnb in a city. This powers the ML models and analytics dashboard.

**County assessor data** enriches the training data with physical property attributes (sqft, year built, lot size) from public tax assessor records. Assessor records are spatially matched to Airbnb listings using KD-tree nearest-neighbor search within a ~200m radius, disambiguated by bedroom count.

| City | Source | Match Rate | Features Added |
|------|--------|:----------:|----------------|
| Seattle | King County Assessor | 85% | sqft, year_built, lot_sqft, bedrooms |
| Austin | Travis CAD | 88% | sqft, year_built, bedrooms, bathrooms |
| Dallas | Dallas CAD | 82% | sqft, year_built, lot_sqft, bedrooms |
| Others | Not available | -- | Features filled with training medians |

The enrichment pipeline (`scripts/enrich_listings.py`) downloads assessor CSVs, queries county ArcGIS services for parcel coordinates, performs spatial matching, and outputs an `assessor_enrichment.csv` per city that the training pipeline automatically picks up.

**For-sale listings** are fetched from Redfin's GIS API (`scripts/download_redfin.py`), which provides up to 10,000 active listings per city. The pipeline filters out vacant land, commercial properties, multi-family 5+ units, and listings without a valid price.

**Street View photos** are fetched from the Google Maps Static API for properties scoring above 30. Images are cached permanently to disk so each location is only fetched once. A monthly counter enforces the free-tier limit (~25,000 requests/month). The API key stays server-side -- the frontend loads images through a proxy route.

## Machine Learning Pipeline

CapScout trains **two XGBoost models per city** from the Inside Airbnb dataset:

- **Price Model** -- predicts the expected nightly rate from property features and location
- **Occupancy Model** -- predicts monthly occupancy with seasonal awareness (uses calendar data to capture month-by-month booking patterns)

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

**Data cleaning notes:** Calendar prices are capped at P95 (95th percentile) rather than P99 because some cities (notably Nashville) have tens of thousands of entries at $9,999 -- hosts use placeholder pricing to block dates. P95 removes this garbage while keeping legitimately expensive listings.

**No hold-out split.** The model trains on 100% of Airbnb data. Cross-validation folds handle evaluation during hyperparameter search, and the final model is retrained on all data with the best parameters. The real "test set" is Redfin for-sale properties -- data the model has never seen and that comes from a completely different distribution.

### Prediction Strategy: Area Median Targeting

The core challenge in STR revenue prediction is that individual Airbnb hosts set wildly different prices for similar properties -- two identical 3-bedroom homes on the same street can list at $150/night and $400/night depending on the host's pricing strategy, photo quality, and marketing. Trying to predict individual host pricing from property features alone yields ~54% accuracy.

CapScout solves this with **area median targeting**: instead of training the model to predict what each individual listing charges, we train it to predict what a **typical listing in that area earns**. The training pipeline:

1. **Clusters** all Airbnb listings into ~20 geographic zones using k-means on coordinates
2. **Groups** listings by (zone, bedroom count, month) -- e.g., "3-bedrooms in East Austin in July"
3. **Replaces** each listing's individual price/occupancy with its group's **median** as the training target
4. The model learns: *"a typical 3BR in this neighborhood earns $X/night in July"*

This eliminates host-specific noise from the training signal. For occupancy, this also mitigates the problem of hosts blocking calendar dates (which looks identical to zero bookings in the data) -- area medians smooth out individual host availability patterns.

For investors, this answers: *"What would a typical STR at this location earn?"* -- not *"What does one specific Airbnb host happen to charge?"*

### Feature Selection (17 features + 2 seasonal)

Only features that exist for **both** Airbnb training data and Redfin for-sale properties are included. Amenities (Redfin has no amenity data), review scores, host metrics, and booking activity were all removed -- they get median-filled as constants for for-sale properties, contributing noise without signal.

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
| `property_age` | 2026 - year_built | derived |
| `lot_acres` | lot_sqft / 43560 | derived |
| `sin_month` | month (1-12) | seasonal |
| `cos_month` | month (1-12) | seasonal |

Physical features (sqft, year_built, lot_sqft) are the key bridge between training data and for-sale properties: Redfin listings already have these, and now the training data does too (via county assessor enrichment). For enriched cities, the model uses real structural data instead of training medians.

### XGBoost + Hyperparameter Tuning

**XGBoost** is the sole model -- appropriate for our data sizes (40K-156K rows per city), handles missing values natively (important for optional features like sqft in unenriched cities), and is histogram-based for fast training.

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

Training reports two R-squared metrics per target:
- **Train R-squared** -- predict on all training data with the final refitted model. Measures how well the model fits the data.
- **CV R-squared** -- average R-squared across the 5 held-out CV folds. Measures generalization to unseen data.

| City | Price CV R-squared | Occ CV R-squared | Notes |
|------|:----------:|:---------:|-------|
| Honolulu | 0.993 | 0.996 | Best overall |
| San Diego | 0.996 | 0.971 | |
| Fort Lauderdale | 0.981 | 0.943 | |
| Las Vegas | 0.980 | 0.980 | |
| Dallas | 0.973 | 0.853 | Assessor enriched |
| New Orleans | 0.972 | 0.947 | |
| Austin | 0.971 | 0.936 | Assessor enriched |
| Asheville | 0.965 | 0.967 | Small but clean |
| Denver | 0.961 | 0.925 | |
| Washington DC | 0.956 | 0.925 | |
| Seattle | 0.951 | 0.920 | Assessor enriched |
| Boston | 0.932 | 0.907 | |
| Nashville | 0.921 | 0.933 | Fixed with P95 cap |
| Fort Worth | 0.886 | 0.809 | Smallest dataset |

### How Predictions Work

When you view a for-sale property, CapScout:

1. Extracts property features (bedrooms, bathrooms, sqft, year_built, lot_size, property type, location)
2. For enriched cities, uses actual sqft/year_built/lot_size from Redfin; for others, fills with training medians
3. Predicts nightly rate and occupancy for each of the 12 months (seasonal via sin/cos encoding)
4. Computes annual revenue from the monthly predictions
5. Runs the financial model (mortgage, taxes, insurance, expenses) to produce the CapScout Score

### Prediction Confidence

Confidence is based on geographic density of training data near the property:
- **High** -- 50+ Airbnb listings within 3km (strong training signal)
- **Medium** -- 15-49 listings within 3km
- **Low** -- fewer than 15 listings (predictions less reliable)

By default, the dashboard hides low-confidence properties. This can be toggled with the "Confident Predictions Only" filter.

## The CapScout Score

The score (0-100) measures **how well a property's predicted STR income covers your ownership costs** -- mortgage payments, property taxes, insurance, and operating expenses. It does not score rental income alone; it reflects overall investment viability.

Five financial metrics are scored on a curve and weighted:

| Component | Max Points | What It Measures |
|-----------|-----------|------------------|
| Gross Yield | 25 | Annual revenue / purchase price |
| Cash-on-Cash ROI | 25 | Annual cash flow / down payment |
| Cap Rate | 20 | Net operating income / purchase price |
| DSCR | 20 | Debt service coverage ratio (NOI / debt service) |
| Occupancy | 10 | Predicted booking rate |

**Score interpretation:** 70+ Excellent (purple) | 55-69 Good (green) | 45-54 Fair (yellow) | <45 Poor (red)

Financial modeling assumes 20% down payment, 7% mortgage rate, 30-year term, and city-specific property tax / insurance / expense rates. All financing assumptions are adjustable via dashboard filters.

## Key Features

- **Dashboard** -- paginated, filterable property list with scores, revenue predictions, and occupancy estimates
- **Interactive Map** -- MapLibre GL map with color-coded score markers and sidebar filters
- **Analytics** -- market overview with neighborhood rankings, price distributions, bedroom breakdowns, and revenue charts
- **Property Detail** -- ML predictions, financial analysis, comparable listings, Street View photo, and confidence scoring
- **Favorites** -- save properties to a local favorites list (persisted in localStorage)
- **Compare** -- side-by-side comparison of up to 3 properties across all metrics
- **Sensitivity Analysis** -- client-side sliders to test how changes in occupancy, nightly rate, and interest rate affect the score
- **Feature Importance** -- see which property features drive price and occupancy predictions (XGBoost feature importances)
- **Prediction Process** -- step-by-step explanation of the full pipeline from data collection through scoring
- **Settings** -- configure API keys and test the prediction model with custom inputs

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
GOOGLE_MAPS_API_KEY=your_key_here
EOF

# Download Inside Airbnb data for a city
python -m scripts.download_data austin

# Download Redfin for-sale listings
python -m scripts.download_redfin austin

# (Optional) Enrich training data with county assessor records
python -m scripts.enrich_listings

# Train ML models (~20 min per city)
python -m models.train_model austin

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
| `GOOGLE_MAPS_API_KEY` | [Google Cloud Console](https://console.cloud.google.com/) | Street View property photos |

The API key is optional. Without it, the app works fully -- Street View photos just won't appear.

## Deployment

- **Frontend**: Deployed to Vercel (auto-deploys from GitHub)
- **Backend**: Deployed to Render (gunicorn with `--preload --workers 1`)
- Model files are compressed with joblib compress=3 to fit within Render's free-tier limits
- Warmup is skipped on Render; models lazy-load on first request per city

## Tech Stack

**Backend:** Python, Flask, scikit-learn, XGBoost, pandas, NumPy, SciPy, joblib, gunicorn

**Frontend:** React 19, Vite, MapLibre GL, Recharts, Axios, React Router
