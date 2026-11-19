CapScout is a personal project built in spare time for learning. It is not a commercial product.

# What the nightly-rate model actually does

This is a study guide, not a product pitch. The question is the same one as the last round: given a public Airbnb listing in one city, how close can we get to the nightly price on that listing, without training on the house next door or on the same host?

The short answer is on a new spatial holdout, seed 23, frozen before any model in this round was fit and scored once at the end. The point model is the v3 blend. Nothing in this round beat it by the pre-set bar.

| City | Test R² | Test MAE | Test RMSE | MAPE | Median APE | 80% coverage | Test listings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin | 0.604 | $61.85 | $125.89 | 33.3% | 23.6% | 81.4% | 974 |
| San Diego | 0.545 | $56.64 | $135.34 | 26.7% | 19.4% | 80.1% | 391 |
| Seattle | 0.667 | $33.92 | $50.98 | 25.2% | 21.5% | 78.9% | 266 |

R² is the share of price variance the predictions capture. MAE is the average absolute miss, in dollars per night. RMSE punishes a few large misses more than MAE does. MAPE is the average of |miss| / price. Median APE is the same idea for a typical listing. Coverage is the share of test listings whose advertised price fell inside the 80% interval.

Those numbers are one split. They are not a better model than v3. On the same rows, a refit of the published v3 specification matches this table exactly, because this round did not change the point model. Seed 23's cells happened to be an easier dollar problem than seed 7, and a smaller one. San Diego's holdout is 391 listings. Seattle's is 266. Do not put $61.85 next to v3's $83.95 and call it an improvement.

The code is `python -m evaluation.run_eval`. The raw output is `reports/metrics.json`, `reports/final_predictions.csv`, `reports/locked_test_predictions.csv`, and `reports/confirmation_predictions.csv`. Sources are in `DATA.md`. The reading for this round is `docs/RESEARCH.md`.

## Against the neighbourhood median, on these rows

A neighbourhood-by-room-type median, fit on the training side of this same split:

| City | Baseline MAE | Model MAE | Error removed |
| --- | ---: | ---: | ---: |
| Austin | $98.98 | $61.85 | 37.5% |
| San Diego | $105.07 | $56.64 | 46.1% |
| Seattle | $53.31 | $33.92 | 36.4% |

"Error removed" is (baseline MAE − model MAE) / baseline MAE. The baseline is weak on a spatial split, because a held-out neighbourhood often falls back to a citywide median. The percent is the gap to that baseline, not a claim that the model is a quote.

On the same rows, size and location only (XGBoost, depth 4) is MAE $74.56, $66.65, and $39.51. Ridge on v3's columns is $69.81, $64.66, and $36.67. The tree blend is ahead of both. That was already true in v3. This split does not change the ranking.

## Seed 7, which is the published comparison

Seed 7 was the locked test in v2 and v3. It had been read, so it was not used to choose anything. The point model here is the v3 model, refit in this code on the same training rows. It matches the published v3 scores exactly.

| City | v3 R² | v3 MAE | This round R² | This round MAE | MAPE | 80% coverage | v3 coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin | 0.451 | $83.95 | 0.451 | $83.95 | 32.0% | 74.9% | 73.2% |
| San Diego | 0.675 | $99.52 | 0.675 | $99.52 | 29.8% | 76.8% | 77.1% |
| Seattle | 0.526 | $38.75 | 0.526 | $38.75 | 27.0% | 80.4% | 83.2% |

The dollars did not move. The coverage moved because the interval method changed, which is the only choice this round kept. Austin went from 73.2% to 74.9%. That is still short of 80%. San Diego went from 77.1% to 76.8%. Seattle went from 83.2% to 80.4%, closer to the target and no longer the wide one. Against the same neighbourhood median as v3 (MAE $127.68, $187.07, $61.90), the model removes 34.2%, 46.8%, and 37.4% of the error. That percent is v3's, because the predictions are v3's.

Seed 11, the v3 confirmation holdout, is the same story. The dollars match the published confirmation scores (MAE $62.36, $70.95, $45.29). Coverage is 83.5%, 78.7%, and 77.8%. Seattle's check is 216 listings. It is too small to hang a claim on.

## Why this test was not used to choose anything

Seed 7 and seed 11 had both been scored and read. Using either of them to pick a feature would have been tuning on a test that was already open. Seed 23 holds out 20% of the cells that are in neither. The ids are in `evaluation/final_ids.json`. They contain no prices. The seed was not searched. It is not 0, 1, 2, 3, 4, 7, 11, or 42.

| City | Final listings | Share of the city | Training rows used |
| --- | ---: | ---: | ---: |
| Austin | 974 | 9.5% | 6,783 |
| San Diego | 391 | 3.4% | 10,283 |
| Seattle | 266 | 4.4% | 5,270 |

The selection folds are five spatial seeds, 0 through 4, on the cells in none of the three holdouts. Mean MAE is an unweighted average of the city-seed scores, so a big city does not outvote a small one. Early stopping is a further spatial slice inside the training side.

Development results for the model that was kept, which is the v3 blend, mean ± standard deviation across those five seeds:

| City | R² | MAE |
| --- | ---: | ---: |
| Austin | 0.547 ± 0.062 | 68.94 ± 11.63 |
| San Diego | 0.678 ± 0.075 | 63.49 ± 19.88 |
| Seattle | 0.454 ± 0.102 | 40.63 ± 6.00 |

San Diego's development MAE still moves by about $20 from seed to seed. One split's MAE is not a stable law. That is also what the final holdout is.

## What was tried, and what was kept

v3's specification was not reopened. It is still size, room type, coordinates, distance to downtown and the airport, census tract income, rent, home value, and density, plus amenities, reviews, host fields, beds, a shared bath, host operations, and the separate review scores other than value. The model is still an equal-weight blend of CatBoost and XGBoost on log price, depth 4. The bar for anything new was the same 1% of mean development MAE.

| Candidate, on the v3 columns | Mean development MAE | Kept? |
| --- | ---: | --- |
| v3 specification | $57.69 | the starting point, and the point model |
| Matched comps, same room type and bedroom bin, within 15 km | $59.43 | no |
| Sentence embedding, eight components | $57.80 | no |
| Earlier Inside Airbnb snapshots, de-duplicated | $57.20 | no, about 0.8%, short of 1% |
| Tail weights, three times the weight at the 90th percentile | $60.19 | no |
| LightGBM as a third equal-weight model | $57.89 | no |
| Learning rate 0.03, on seeds 0–1 | $59.65 against $59.47 | no |
| Learning rate 0.08, on seeds 0–1 | $60.20 against $59.47 | no |
| Minimum child weight 20, on seeds 0–1 | $59.35 against $59.47 | no, about 0.2% |
| Non-negative blend, weights fit on seeds 0–1, scored on seeds 2–4 | $56.39 against $56.50 | no, about 0.2% |
| Duan smearing, cross-fit | $60.61 | no |
| Isotonic map from predicted dollars to actual dollars, cross-fit | $62.26 | no |

Per city, mean ± standard deviation of MAE. "Selected" matches v3 because nothing was added.

| Set | Austin | San Diego | Seattle |
| --- | ---: | ---: | ---: |
| v3 specification | 68.94 ± 11.63 | 63.49 ± 19.88 | 40.63 ± 6.00 |
| Matched comps | 70.61 ± 12.35 | 64.80 ± 20.85 | 42.88 ± 9.43 |
| Sentence embedding | 69.58 ± 12.70 | 63.33 ± 19.79 | 40.48 ± 5.72 |
| Earlier snapshots | 69.42 ± 11.70 | 63.33 ± 18.06 | 38.85 ± 6.22 |
| Tail weights | 70.69 ± 10.32 | 66.21 ± 18.85 | 43.65 ± 8.22 |
| Selected | 68.94 ± 11.63 | 63.49 ± 19.88 | 40.63 ± 6.00 |

The chart is `reports/figures/ablation_mae.png`. Read the differences. The embedding is a few cents in the average and a dollar in San Diego and Seattle, the wrong direction in Austin. The earlier snapshots help Seattle by about $2 and do not help Austin, which is the city that had a year of them. The average of those three cities is the rule, and it missed 1%. Comps and tail weights are worse in every city.

The hyperparameter screen was not allowed to win on the seeds that picked it. Minimum child weight 20 was the best of the three and was still short of 1% on seeds 0 and 1, so seeds 2–4 were not asked to confirm it. The weighted blend was fit on seeds 0 and 1 only. It gave CatBoost about three quarters of Austin, XGBoost about 85% of San Diego, and all of Seattle. On the seeds it had not seen, the gain was about 0.2%. Equal weights stayed.

## The earlier snapshots

Inside Airbnb's public page keeps about the last twelve months. Files dated after the study snapshot were not used. A 2026 price is not training data for a 2025 label. What was still hosted, and earlier than the study file, was Austin monthly from 2024-04-21 through 2025-05-11, and 2025-02-18 for San Diego and for Seattle.

A listing already in the study file was not repeated. That left 9,113 Austin listings, 898 San Diego listings, and 429 Seattle listings. They can enter training only. A row in a held-out cell, a row whose host has a held-out listing, and a row within 1 km of a held-out listing are dropped. During selection, rows in the seed-23 cells were dropped as well, so the development score was not using the past prices of the blocks it was not allowed to see.

The gain was real in Seattle and absent in Austin. One month of Seattle listings that had left the file by the study date was worth about $2 of development MAE. A year of Austin listings that had left the file was not. The protocol averages the cities. 0.8% is not 1%.

## Prediction intervals

The interval in the app is an 80% band. v3 scaled one log residual by the predicted price. That band was still thin in some price bands. This round compared three methods on the development folds only:

| Method | Worst-city coverage | Worst-city coverage in the worst price quintile | Mean of city median widths |
| --- | ---: | ---: | ---: |
| Log residual, width scaled with the predicted price | 80.0% | 64.3% | $123 |
| Asymmetric log residual, one pair of quantiles | 79.7% | 67.4% | $128 |
| Asymmetric, a separate pair in each predicted-price quintile | 79.7% | 71.7% | $118 |

The rule, written down first, keeps the scaled band unless a challenger covers the worst quintile at least 3 points better, without dropping the worst city by more than a point, without getting more than 35% wider, and without leaving the point estimate outside the band. Both challengers cleared the 3 points. The quintile version was the better of the two, and narrower than the scaled band, so it is the band in the app. A quintile with fewer than 40 calibration residuals uses the city-wide quantile. The test was not used to pick.

On the final holdout that band covered 81.4% of Austin (median width $120), 80.1% of San Diego ($107), and 78.9% of Seattle ($76). The same holdout, scored once with the other two methods and not used to switch:

| City | Scaled band | Asymmetric | By price quintile (the one in the app) |
| --- | ---: | ---: | ---: |
| Austin | 81.4% | 80.9% | 81.4% |
| San Diego | 78.5% | 81.6% | 80.1% |
| Seattle | 82.7% | 84.6% | 78.9% |

On this split the three methods are close, and Seattle's quintile band is the one under 80%. The hole is not only the top of the market. Seattle's band around $127 to $150 was covered 60.6%. San Diego's band around $67 to $85 was covered 67.3%, and the top eighth, roughly $277 to $975, was covered 69.4%. Austin's weakest eighth, roughly $148 to $195, was covered 72.7%. Development coverage of the chosen band was 80.2% in Austin, 79.7% in San Diego, and 82.0% in Seattle. The holdout did not match that in Seattle. A held-out part of the city is not an exchangeable draw, so the conformal guarantee does not fully apply. The quantiles were not widened after seeing 78.9%.

On seed 7, which is the comparison that can be put next to v3, the new band covered 74.9%, 76.8%, and 80.4%. Austin is better than 73.2% and it is not 80%. The development rule improved the worst quintile. It did not finish the job on the published test.

## Where the errors are

Dollar error still grows with the predicted price, and expensive listings are still under-predicted. On the final holdout, Austin's top fifth of predictions, roughly $229 to $1,346, has MAE about $154 and a mean residual of about +$18. San Diego's top fifth, roughly $238 to $975, has MAE about $150 and a mean residual of about +$76. Seattle's top fifth has MAE about $58 and almost no bias. The tail fixes that were supposed to move this — smearing, isotonic regression, and tail weights — made the average absolute error worse on the development folds, so they were not applied here. The under-prediction is still in the table.

The chart of predicted versus actual is `reports/figures/pred_vs_actual.png`. Residuals are `reports/figures/residuals.png`. Coverage by predicted price is `reports/figures/calibration.png`.

## What the model is using

SHAP values are the average absolute contribution to the log-price prediction, on a sample of up to 600 final-holdout rows, averaged across the two models. They explain the model we scored. They were not used to drop features. The chart is `reports/figures/shap.png`.

- **Austin:** capacity (0.15), bathrooms (0.15), reviews per month (0.11), bedrooms (0.10), minimum nights (0.09), tract home value (0.08).
- **San Diego:** capacity (0.25) and longitude (0.14). Longitude is the coast. Then bedrooms, bathrooms, and reviews per month.
- **Seattle:** bedrooms (0.13), capacity (0.09), distance to downtown (0.09), reviews per month (0.06).

That is the same story as v3. No new column entered the model, so no new column should lead the chart.

## The old R² near 0.97

The previous training target, in a pipeline this study does not use, was not the listing's price. Listings were clustered geographically, grouped by cluster and bedroom count, and each row's target was replaced by that group's median price. The model was then scored against those medians.

On Austin, as a demonstration and not as a result (it uses the whole city, including what would have been a test):

- A model trained on the group median and scored against that median has cross-validated R² **0.973** and MAE about **$7**.
- The same predictions, scored against the listing's own price, have R² **0.385** and MAE **$84.40**.

That is the inflation. This study does not train on a median.

## What we tried that did not help

**Matched comps.** Median log price of the training listings with the same room type and the same bedroom bin, inside 15 km, at least eight of them, own price removed. Mean development MAE went from $57.69 to $59.43. v2's 40-neighbour median and v3's 5 km kernel had already failed under the 1 km buffer. A fourth version, matched on the product and given more room, failed by more. The buffer is doing what it was built to do. The places over there are not a comp for a held-out cell.

**A sentence embedding.** MinishLab's potion-base-8M, a small static model, compressed to eight components with a PCA fit inside the training fold. Eight was the width of the TF-IDF that failed in v3. Mean MAE went from $57.69 to $57.80. The ad is mostly size, a room type, and an amenity the structured columns already have. Photos were not downloaded.

**Earlier snapshots.** Described above. 0.8%, and the city that gained was not the city with the extra year of data.

**Tail weights, smearing, and isotonic regression.** Absolute error wants the median. Up-weighting expensive listings, and pushing the prediction toward the mean with Duan's smearing factor, both made MAE worse. An isotonic curve fit on out-of-fold predictions in one block of the city did not correct the next block. The expensive-listing bias is real. These three fixes are not the way through it under this split.

**LightGBM, a three-point grid, and a weighted blend.** LightGBM as a third equal model was slightly worse. The grid's best member was 0.2% better on the seeds that ranked it, which was not the bar, so it was not confirmed on the other seeds. The weighted blend was 0.2% better on the seeds that had not fit the weights. Equal CatBoost and XGBoost stayed. Optuna was not run. A search that adapts to all five seeds would have been another chance to fit those seeds.

## Every cleaning step

Cleaning is in `data/ingest.py`. The rules are fixed dollar amounts, not percentiles of the file. A percentile computed before the split would let the test listings help decide who counts as an outlier. The same rules are applied to the earlier snapshots.

1. Keep every room type. Room type is a feature, and the baseline also gets room type.
2. Parse the price. Drop missing or non-positive prices.
3. Drop prices under $25 and over $2,000. The upper cut removes the row. It does not quietly rewrite a $5,000 night as $2,000.
4. Drop rows with no latitude, longitude, or room type.
5. Drop rows with no bedroom count, or a bedroom count outside 0–10. Zero is a studio and stays.
6. If guest capacity is missing or zero, fill it with two guests per bedroom. Observed capacities are left alone.

| Rule | Austin | San Diego | Seattle |
| --- | ---: | ---: | ---: |
| Rows in the file | 10,533 | 12,991 | 6,772 |
| Missing or non-positive price | 16 | 1,287 | 764 |
| Price under $25 | 72 | 3 | 2 |
| Price over $2,000 | 122 | 88 | 2 |
| Missing bedrooms | 13 | 19 | 10 |
| Bedrooms outside 0–10 | 14 | 4 | 2 |
| Rows used | 10,296 | 11,590 | 5,992 |

Square footage is not on these files. This study does not invent it.

## How the data was split

**A cell, not a row.** Each listing is assigned to a grid cell of 0.03 degrees, about 3 km of latitude, from its coordinates alone.

**Seed 7** holds out 20% of cells. The ids are in `evaluation/locked_test_ids.json`. They were written from coordinates and the seed. The code refuses to regenerate them if they no longer match. Those listings are the secondary comparison above.

**Seed 11** holds out 20% of the cells that remain. The ids are in `evaluation/confirm_ids.json`.

**Seed 23** holds out 20% of the cells that remain after that. The ids are in `evaluation/final_ids.json`. The code refuses to regenerate them too.

**Tuning never sees any of the three.** The selection folds are seeds 0 through 4 of what is left.

**The target is log(1 + price).** Every reported metric is computed after converting predictions back to dollars.

## Self-audit

Checked in code and recorded in `reports/metrics.json`:

- The frozen listing ids match seeds 7, 11, and 23 recomputed from coordinates. The three lists do not overlap. None of the files contains a price.
- No spatial cell is in both training and test, for the final holdout, the locked test, or the confirmation split. Listings that sat in a final-holdout cell but had already been dropped from an earlier pool are not put back into training.
- No host is in both sides. No training listing is within 1 km of a held-out listing.
- The scored columns are v3's. They do not include a price, a value score, availability, or an estimated revenue.
- Add-ons, the third library, the grid, the weighted blend, the corrections, and the interval were chosen by the functions in `evaluation/select.py` from development MAE or development coverage only. `tests/test_report.py` replays those functions on the JSON and fails if the recorded winner was edited. The same test replays v2's and v3's archived choices.
- Seed 7's dollar metrics match `reports/v3_metrics.json`. The point model did not move.
- The final-holdout metrics match `reports/final_predictions.csv`. The seed-7 metrics match `reports/locked_test_predictions.csv`. The seed-11 metrics match `reports/confirmation_predictions.csv`.

`audit_passed` in the JSON is true.

## What the local app is doing

`streamlit run app/streamlit_app.py` serves `127.0.0.1` only. It is a way to read one prediction and the test report. It is not a second evaluation and it is not deployed.

- The nightly rate is the v3 blend. This round did not replace it.
- The band is the quintile interval from this round.
- The reasons are SHAP values on that row, turned into a sentence and an approximate percent.
- Nearby rates are Inside Airbnb advertised prices. They are not inputs to the model.
- An address is geocoded with the Census geocoder. The map can also take a click.
- Description, amenities, reviews, and superhost status are optional. Left blank, they stay missing. Missing is not "no kitchen."

## Limitations

- The label is the advertised rate, not what a guest paid and not what a host cleared after fees.
- The point model did not improve on v3. The final-holdout MAE is lower because the cells are different, and fewer. That is the result, not a rounding choice.
- Austin's 80% interval covered 74.9% of seed 7. It covered 81.4% of seed 23. Both are one split.
- Seattle's final holdout covered 78.9%, and one price band covered 60.6%. The quantiles were not edited after that look.
- San Diego's final holdout is 391 listings. Seattle's is 266. One unlucky set of cells moves those R² figures.
- Expensive listings are still under-predicted, especially in San Diego. The three tail fixes that were tried made average absolute error worse.
- Airbnb jitters coordinates by up to about 150 meters. Distances, cells, and the tract join are fuzzy at that scale.
- The 1 km buffer and the host purge throw away a large share of training rows.
- Development seed-to-seed MAE in San Diego still moves by about $20.
- The saved model has not seen the seed-23 listings. That is intentional. It has seen the seed-7 and seed-11 listings, because those are part of the training population for the final-holdout model. The seed-7 score itself uses a different fit that has not seen them.
- Three cities, one protocol. Nothing here is a national model.

## How to reproduce

From the repository root, with the pinned `requirements.txt`:

```bash
pip install -r requirements.txt
python -m evaluation.run_eval
pytest
```

The evaluation refits the development splits, writes the report, and scores seed 23 once. It then scores seed 7 and seed 11 once, as secondary comparisons. It does not reopen v3's feature search. v3's published metrics stay in `reports/v3_metrics.json`.

The local app, after that:

```bash
streamlit run app/streamlit_app.py
```
