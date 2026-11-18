CapScout is a personal project built in spare time for fun and learning. It is not a commercial product.

# CapScout

A study of how well a public Airbnb listing’s advertised nightly rate can be predicted from the listing itself, without leaking nearby listings or the same host into the test set.

The write-up is [reports/FINDINGS.md](reports/FINDINGS.md). That file is the study guide: where the data came from, every cleaning rule, why each feature is there, how the split works, what was tried, and what the numbers mean. This page is the short version.

## The result

One locked spatial test per city, scored once. The model is XGBoost on log price, depth 4, using size, room type, coordinates, distance to downtown, and distance to the airport. Metrics are in dollars, after converting back from the log.

| City | Test R² | Test MAE | Test RMSE |
| --- | ---: | ---: | ---: |
| Austin | 0.426 | $95.61 | $164.78 |
| San Diego | 0.555 | $85.02 | $170.79 |
| Seattle | 0.437 | $49.80 | $95.16 |

On the same rows, a neighbourhood-by-room-type median is worse, and in Austin it is worse than predicting the average (test R² −0.688). A ridge regression on the same columns lands at 0.026 / 0.461 / 0.347. The tree model wins the locked test in all three cities. San Diego’s development splits actually preferred ridge on MAE; that is reported in the findings and was not used to change the test.

An older version of this project trained on area-median prices and reported R² near 0.97. On Austin, a model scored against that smoothed target still shows R² 0.973, and the same predictions scored against the listing’s own price show R² 0.385. The high number was the target being easy. It is not a result of this study.

## What was deliberately left out

Only Austin, San Diego, and Seattle are modeled. They are single metros, the snapshots are from 2025, and bedrooms are filled in. Hawaii is several islands in one file. Several other cities are missing a price on about 30% of rows. Asheville’s bedrooms column is empty. Those files remain on disk and are not in the headline table.

Listing quality (reviews, amenities, host flags) and nearest-neighbour prices were measured on the development splits. Neighbour comps got worse once listings within 1 km were removed from training. Reviews and amenities helped Austin and hurt Seattle, and the average gain was under a pre-set 1% bar, so they are not in the model. LightGBM was about 1% better on the development average, short of a pre-set 2% bar, so the model stays XGBoost.

## Data

Inside Airbnb listings, CC BY 4.0. Credit Inside Airbnb. The snapshots in `data/raw/cities/` are filtered; the license and the exact files are described in the findings. This project does not scrape Airbnb.com, and it does not download listing-site inventories.

Occupancy in the local app, if you run it, is a neighbourhood median of Inside Airbnb’s published review-based proxy (the “San Francisco model,” capped at 70%). It is not observed bookings and it is not a trained model.

## Layout

| Path | What it is |
| --- | --- |
| `data/` | City list, download, cleaning, the occupancy proxy |
| `features/` | Columns that do not use the target, and fold-safe price encodings |
| `evaluation/` | The split, the baselines, the tree models, the report |
| `models/` | The saved development-set artifacts and the scorer |
| `reports/` | Findings, metrics, and figures |
| `backend/`, `src/` | A small local app for looking at the three cities |

## Run it

```bash
pip install -r requirements.txt
python -m evaluation.run_eval    # rewrites reports/ and models/artifacts/
pytest

# local app, after the artifacts exist
cd backend && python app.py
# other terminal, from the repo root
npm install && npm run dev
```

`evaluation/protocol.py` is the design: the test seed, the 1 km buffer, the host rule, and the bars for “simpler model” and “switch to LightGBM.” Change those and you are running a different study.

Tests check the split, the occupancy formula, that a fold encoder cannot see a price it was not fit on, and that the figures in this README and in the findings match `reports/metrics.json`.
