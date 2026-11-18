CapScout is a personal project built in Corbin's spare time, for learning and as a portfolio piece. It is not a commercial product.

# CapScout

How close can a public model get to an Airbnb listing's advertised nightly rate, without training on the listing next door or on the same host?

The long version is [reports/FINDINGS.md](reports/FINDINGS.md). Sources and licenses are in [DATA.md](DATA.md).

## The result

One locked spatial test per city, frozen before the new features were scored, and scored once. The model is an equal-weight blend of CatBoost and XGBoost on log price. It uses size, room type, coordinates, distance to downtown and the airport, census tract income, rent, home value, and density, and the listing's own amenities and reviews. Metrics are in dollars.

| City | Test R² | Test MAE | Test RMSE | MAPE | Median APE | Listings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin | 0.422 | $87.74 | $160.46 | 33.6% | 29.0% | 2,796 |
| San Diego | 0.678 | $98.96 | $178.04 | 30.4% | 24.0% | 875 |
| Seattle | 0.532 | $38.19 | $74.44 | 26.6% | 21.1% | 2,042 |

On the same rows, a neighbourhood-by-room-type median is worse (MAE $127.68, $187.07, $61.90). The previous specification, size and location only, lands at R² 0.344 / 0.523 / 0.458. A ridge regression on the selected columns lands at 0.258 / 0.519 / 0.464. Development R² for the selected model was 0.562 ± 0.054 in Austin, 0.568 ± 0.070 in San Diego, and 0.544 ± 0.084 in Seattle.

An earlier study on a different spatial split, with a smaller feature set, reported R² 0.426, 0.555, and 0.437. Those numbers are not this test. The comparison that uses the same listings is the "previous specification" row in the findings.

An older pipeline trained on area-median prices and showed R² near 0.97. That was the target being easy. Scored against the listing's own price, the same exercise is about R² 0.385. It is a demonstration, not a result of this study.

## What did not help

OpenStreetMap distances and counts, building-type flags, smoothed neighbourhood prices, nearest-training-listing prices, and monotone constraints did not clear the pre-set bar. CatBoost alone was a little better than XGBoost and not better enough to replace it; it is half of the blend because the blend was. Detail is in the findings. The 80% intervals covered 66.7% of Austin's test, 75.2% of San Diego's, and 86.6% of Seattle's. They are not uniformly calibrated.

## Data

| Source | License | Used for |
| --- | --- | --- |
| Inside Airbnb listings | CC BY 4.0 | The rows, the price, size, room type, amenities, reviews |
| Census ACS 2024 5-year and TIGER/Line tracts | Public domain | Tract income, rent, home value, density |
| OpenStreetMap via Overpass | ODbL, © OpenStreetMap contributors | Measured, not in the scored model |

The calendar file was not used. Unavailable is not the same thing as booked. Redfin, Zillow, and AirDNA were not used.

## Reproduce

From the repository root, with the pinned `requirements.txt`:

```bash
pip install -r requirements.txt
python -m evaluation.run_eval
pytest
```

That command downloads the Census and OpenStreetMap extracts if `data/external/` is missing, rebuilds features, fits the development splits, writes `reports/metrics.json`, and scores the locked test once. The first download is about 150 MB of Census files plus a few Overpass queries. Cached city extracts are already in the repo, so a second run skips the network. The evaluation itself is a few minutes on a laptop.

`evaluation/protocol.py` is the design: the test seed, the 1 km buffer, the host rule, and the bars for a simpler feature set, a new library, and the blend. Change those and you are running a different study. The held-out listing ids are in `evaluation/locked_test_ids.json`.

## Run the local app

After the artifacts exist (they are committed, or they appear when the command above finishes):

```bash
streamlit run app/streamlit_app.py
```

The app binds to `127.0.0.1:8501` only. Pick a city, search an address or click the map, enter bedrooms, bathrooms, guests, and room type. It returns a nightly rate, an 80% interval, the main reasons, and nearby advertised rates. A second page shows the locked-test metrics and the figures.

![CapScout local estimate](docs/app-estimate.png)

The rate is an advertised-rate estimate for a listing that looks like the one you described. Reviews are part of the scored model and are left missing when you have not supplied them, which is the right input for a home that has never been listed. It is not a quote.

## Limitations

- The label is the advertised rate, not what a guest paid.
- Errors of tens of dollars, and intervals a hundred dollars wide, are the honest precision.
- San Diego's test is 875 listings. That R² is one split.
- Seattle's training set after the buffer is 1,507 listings, against a test of 2,042.
- Three cities, one protocol. This is not a national model.
