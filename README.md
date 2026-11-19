CapScout is a personal project built in Corbin's spare time, for learning and as a portfolio piece. It is not a commercial product. It runs on your computer. It is not a hosted site.

# CapScout

What nightly rate could a home in Austin, San Diego, or Seattle charge on Airbnb, and about how many nights a year would it book?

The long version is [reports/FINDINGS.md](reports/FINDINGS.md). A one-page way to explain the work out loud is [docs/EXPLAIN_IT_YOURSELF.md](docs/EXPLAIN_IT_YOURSELF.md). Sources and licenses are in [DATA.md](DATA.md).

## The result

The holdout was frozen before any model was fit. Seed 17 holds out about 20% of 3 km grid cells in each city. A host with a held-out listing is removed from training, and so is any training home within 1 km of a held-out home. Seeds 19, 29, and 37 repeat the same frozen choice. They were not used to pick a feature or a model.

The shipped nightly-rate model is ridge regression on the log of the advertised rate. It is one model, not a blend. CatBoost was fit once, at the settings already chosen for XGBoost, and it did not clear a 5% bar that was written down first. Neither did an equal blend of the two. Neither is in the app.

The lowest average error was $54.19 a night, from XGBoost. Ridge regression was $55.18, inside the 2% band (up to $55.28), and it is the simplest model in that band.

CatBoost averaged $54.81 and the equal blend averaged $53.69. Neither beat Ridge regression's $55.18 by the pre-set 5% margin, which would have required $52.42 or less. CatBoost was fit once, at the depth and learning rate already chosen for XGBoost, not on its own search. Neither CatBoost nor the blend is in the app.

Average miss on the primary holdout, in dollars. Each city counts equally.

| Model | Austin, TX | San Diego, CA | Seattle, WA | Average |
| --- | ---: | ---: | ---: | ---: |
| Area-typical rate | $112.75 | $91.62 | $49.40 | $84.59 |
| Linear regression | $73.26 | $56.89 | $36.26 | $55.47 |
| Ridge regression | $72.13 | $57.27 | $36.13 | $55.18 |
| Random forest | $72.09 | $55.98 | $35.92 | $54.67 |
| XGBoost | $71.81 | $53.88 | $36.89 | $54.19 |
| Neural net (comparison only) | $84.85 | $63.41 | $41.53 | $63.26 |
| CatBoost | $74.64 | $53.41 | $36.37 | $54.81 |
| 50/50 XGBoost and CatBoost | $71.35 | $53.42 | $36.31 | $53.69 |

Ridge on the primary holdout. The intervals resample held-out grid cells, not individual listings.

| City | Listings | Cells | R² | R² interval | MAE | MAE interval | Median % error | Median % interval | Area-typical MAE | Error removed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin, TX | 2,101 | 31 | 0.511 | 0.449 to 0.665 | $72.13 | $56.38 to $85.13 | 25.9% | 22.7% to 28.1% | $112.75 | 36.0% |
| San Diego, CA | 817 | 23 | 0.601 | 0.517 to 0.673 | $57.27 | $45.40 to $75.88 | 21.5% | 19.2% to 24.5% | $91.62 | 37.5% |
| Seattle, WA | 730 | 8 | 0.522 | 0.489 to 0.592 | $36.13 | $31.43 to $40.24 | 19.9% | 17.3% to 22.4% | $49.40 | 26.9% |

The same ridge model on three other holdout seeds. The spread is the gap between the highest and lowest of those three, not a new score to quote.

| City | Primary R² | Other seeds R² | Primary MAE | Other seeds MAE |
| --- | ---: | ---: | ---: | ---: |
| Austin, TX | 0.511 | 0.455, 0.396, 0.574 (spread 0.178) | $72.13 | $54.41, $64.01, $49.60 (spread $14.41) |
| San Diego, CA | 0.601 | 0.650, 0.650, 0.182 (spread 0.468) | $57.27 | $69.37, $74.25, $62.31 (spread $11.94) |
| Seattle, WA | 0.522 | 0.465, 0.544, 0.492 (spread 0.079) | $36.13 | $39.79, $36.18, $32.46 (spread $7.33) |

San Diego's R² on those other seeds runs from 0.182 to 0.650. The primary number is not a lucky average of them. Austin's primary dollar error ($72.13) is higher than its other seeds, so the headline Austin figure is the harder draw.

A typical home is off by about 20% to 26%. The dollar error is larger than that percent suggests, because a few expensive listings miss by a lot. The range shown in the app covered 79.0% of Austin's holdout, 82.0% of San Diego's, and 81.8% of Seattle's. Those are the measured coverages. The band was not widened after that look.

## Booked nights

The nights label is not a booking count. It is a review-based estimate: reviews in the last year divided by 0.50, times 3 nights (or a longer minimum stay, up to 90), capped at 255.5 nights. The model cannot see the review count, the minimum stay, the price, or availability.

The lowest average error was 69.0 nights, from Random forest. Linear regression was 69.3 nights, inside the 2% band (up to 70.4 nights), and it is the simplest model in that band.

The nights model did not clear the pre-set bar in every city. The bar was an average miss at least 10% lower than the area-typical nights, and an R² of at least 0.10. The app labels nights as a rough estimate.

| Model | Austin, TX | San Diego, CA | Seattle, WA | Average |
| --- | ---: | ---: | ---: | ---: |
| Area-typical rate | 72.0 | 76.5 | 72.8 | 73.8 |
| Linear regression | 68.9 | 70.2 | 68.7 | 69.3 |
| Ridge regression | 68.6 | 70.3 | 68.8 | 69.2 |
| Random forest | 67.9 | 70.5 | 68.7 | 69.0 |
| XGBoost | 69.0 | 69.9 | 68.7 | 69.2 |

| City | R² | R² interval | MAE (nights) | MAE interval | Median % error | Area-typical MAE | Error removed | Cleared |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin, TX | 0.014 | -0.013 to 0.055 | 68.9 | 64.8 to 70.8 | 54.2% | 72.0 | 4.4% | No |
| San Diego, CA | 0.118 | 0.042 to 0.144 | 70.2 | 66.6 to 71.4 | 39.9% | 76.5 | 8.1% | No |
| Seattle, WA | 0.062 | -0.006 to 0.120 | 68.7 | 64.8 to 70.8 | 40.0% | 72.8 | 5.7% | No |

| City | Primary R² | Other seeds R² | Primary MAE | Other seeds MAE |
| --- | ---: | ---: | ---: | ---: |
| Austin, TX | 0.014 | 0.032, 0.033, 0.057 (spread 0.025) | 68.9 | 68.5, 65.8, 66.6 (spread 2.8) |
| San Diego, CA | 0.118 | 0.100, 0.095, 0.060 (spread 0.039) | 70.2 | 71.2, 70.8, 73.1 (spread 2.3) |
| Seattle, WA | 0.062 | 0.073, 0.063, 0.056 (spread 0.016) | 68.7 | 70.5, 71.7, 71.9 (spread 1.4) |

The nights models predict nights directly. On the training folds a log target averaged 73.76 nights of error against 71.84 for the direct target, which is not a 2% improvement, so the log target was not used.

Yearly revenue in the app is the predicted rate times the predicted nights. On the primary holdout that product has R² 0.266 in Austin, 0.276 in San Diego, and 0.161 in Seattle. The letter grade matched the grade of the review-based revenue about 32%, 31%, and 30% of the time. The grade is a rank inside the city. It is not a profit.

## What it does well, and what it does not

It beats the usual rate for the neighbourhood and room type, by 36.0% of the dollar error in Austin, 37.5% in San Diego, and 26.9% in Seattle. The features are ones a person can name: size, room type, location, a few amenities, and, for the rate only, reviews. Blank fields in the app use a typical training value, not zero.

It does not know what a guest paid. The intervals are wide: about $155 in Austin, $115 in San Diego, and $83 in Seattle at the median. San Diego's R² moves sharply when a different set of cells is held out. The nights number is a rough estimate of a review proxy, and it missed the bar. Three cities are not a national model.

## Data, short version

| City | Rows in the file | Rows kept |
| --- | ---: | ---: |
| Austin, snapshot 2025-06-13 | 10,533 | 7,720 |
| San Diego, snapshot 2025-03-16 | 12,991 | 8,387 |
| Seattle, snapshot 2025-03-16 | 6,772 | 4,610 |

The big cuts are a missing or non-positive price, no review in the last 12 months, and a price outside $25 to $2,000. Nothing in these three files sat more than 60 km from downtown. Duplicate listing ids were already unique. The full funnel, with every row count, is in [reports/data_quality.md](reports/data_quality.md) and in the findings.

Census tract income, rent, home value, and density were joined from the public ACS 2024 5-year tables. Almost every listing landed in a tract. Adding those columns did not cut training-fold error by 2%, so they are not in the model. An earlier round already measured OpenStreetMap places and left them out. This round did not add them back.

An older pipeline trained on area-median prices. As a demonstration, that fit scored about 0.973 against the smoothed median and about 0.385 against the listing's own price. This study trains on the listing's own price.

## Run it yourself

From the repository root:

```bash
pip install -r requirements.txt
python -m data.download austin san-diego seattle
streamlit run app/streamlit_app.py
```

The download pulls the three Inside Airbnb snapshots. They are not stored in git. The app listens on `127.0.0.1` port 8501. Pick a city, search an address or type a latitude and longitude, and enter bedrooms, bathrooms, guests, and room type. You get a nightly rate with a range, nights per year, yearly revenue, a letter grade, the top reasons, and nearby ads. A bad address, a lookup that does not answer, or a pin more than 60 km from downtown is a message, not a score. A purchase price is optional and only produces a rough gross yield.

The saved models are already in `models/artifacts/`. `python -m evaluation.run_study` refits them. It will refuse to run if the frozen holdout ids in `evaluation/study_ids.json` no longer match the cleaned rows.

## Host it yourself

You can run the same app in Docker. This does not publish a site.

```bash
python -m data.download austin san-diego seattle
docker compose up --build
```

Compose publishes the app only on `127.0.0.1:8501`. Inside the container the process listens on all interfaces so Docker can forward that port. The listing files are mounted from `data/raw` so the nearby ads show up. The estimate still runs if that folder is empty.

`vercel.json` keeps a static-page build for the older browser code and sets git deployment to off, so a connected host does not deploy on every push. Do not put a public address in this repo.

`.env.example` lists optional keys for the older Flask API. The Streamlit app does not read them. Leave them blank.

## Tests

```bash
pytest
```

CI runs ruff and black, downloads the three snapshots, then runs pytest.
