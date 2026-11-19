CapScout is a personal project built in spare time for learning. It is not a commercial product.

# What the nightly-rate model actually does

This is a study guide, not a product pitch. The nightly-rate question is unchanged: given a public Airbnb listing in one city, how close can we get to the nightly price on that listing, without training on the house next door or on the same host? The price numbers below are the ones already published. A separate question, booked nights, is at the end. It did not clear its bar.

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

- The nightly rate is the v3 blend. The nights round did not replace it.
- Booked nights, annual revenue, and the A–F grade come from the separate nights model. The grade is the percentile rule in the booked-nights section. A purchase price, if typed, is divided into the predicted revenue and labeled a rough gross yield.
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

## Booked nights

The price model is untouched. This section is a different label, a different seed, and a different bar.

The question is how many nights a home would book in a year. Public calendars cannot answer it. A night marked unavailable is blocked or booked, and the file does not say which. AirDNA-style products try to separate those with data this study does not have. The label here is the Inside Airbnb San Francisco model, applied to the last twelve months of reviews:

Reviews in the last year, divided by a 50% review rate, times a stay of 3 nights. If the listing's minimum stay is longer than 3 and at most 90, that minimum is the stay instead. The result is capped at 255.5 nights, which is 70% of 365. The 50%, the 3 nights, and the 70% cap are Inside Airbnb's published choices. The 3 nights is their fallback when a city has no statement of average stay that this study was willing to treat as a measurement. The hand-set 3.2 and 4.0 figures elsewhere in the city file were not used. A listing with no reviews in the last year is left out. It is not coded as zero nights.

That label is a model of who left a review, scaled into nights. It is not a count of bookings. Guests who do not review are invisible. The minimum stay is part of the formula and is not a feature, so two similar houses with different house rules do not have to get the same label, and the model is not allowed to read the rule. Review counts, review scores, availability, host performance, and price are not features either. The model sees the house: size, room type, location, the census tract, the building type, and amenities.

| City | Listings with a label | Share at the 255.5-night cap | Median nights | Held out | Training rows used |
| --- | ---: | ---: | ---: | ---: | ---: |
| Austin | 7,832 | 12.3% | 90 | 2,420 | 2,577 |
| San Diego | 8,571 | 17.6% | 124 | 2,655 | 2,970 |
| Seattle | 4,696 | 23.6% | 144 | 765 | 2,193 |

The training rows are what remain after the host purge and the 1 km buffer. In Austin and San Diego that remainder is about the size of the holdout. The buffer is doing what it did for price. It is not a bug in the split.

### The holdout and the bar

Seed 31 holds out 20% of the 0.03-degree cells on the listings that have a label. The ids are in `evaluation/occupancy_ids.json`. They contain no nights and no prices. The seed is not 0, 1, 2, 3, 4, 7, 11, or 23, and it was not searched. The file was written before any nights model was fit.

The bar was written down first. A city clears it only when the holdout MAE is at most 90% of the neighbourhood-median MAE and the holdout R² is at least 0.10. Both are required. A 1% feature rule would have called a useless model a win. Development folds, seeds 0 through 4, choose the feature prefix. The holdout is scored once.

The prefix rule is the same 1% simplicity rule as the price study, on log nights, XGBoost, depth 4. Mean development MAE, cities weighted equally:

| Prefix | Mean development MAE |
| --- | ---: |
| Size and room type | 75.45 |
| Plus location | 75.46 |
| Plus census | 75.72 |
| Plus building type | 75.11 |
| Plus amenities and beds | 73.15 |

Amenities cleared 1% against the building-type prefix. Location and census did not help. An unlogged target then cut that development MAE from 73.15 to 70.79, which is more than 1%, so the scored model predicts nights directly rather than log nights. An equal CatBoost and XGBoost blend reached 70.45 against XGBoost's 70.79. That is about half a percent. The blend was not kept. The holdout model is XGBoost, depth 4, on the amenity prefix, in nights.

Adding `number_of_reviews_ltm` on development seed 0, which is not allowed, dropped MAE from 69.9 to 26.4 in Austin, from 79.5 to 27.6 in San Diego, and from 72.2 to 26.3 in Seattle. The column is an input to the label. It was not eligible, and it is not in the model.

### What the holdout said

| City | R² | MAE | RMSE | Median APE | Neighbourhood median MAE | Room-type median MAE | Error removed | Test listings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin | 0.073 | 70.0 | 82.4 | 52.9% | 76.9 | 77.0 | 8.9% | 2,420 |
| San Diego | 0.098 | 72.0 | 82.8 | 45.3% | 78.9 | 79.4 | 8.7% | 2,655 |
| Seattle | 0.078 | 71.6 | 81.6 | 42.2% | 79.5 | 78.5 | 9.9% | 765 |

MAE is nights per year. "Error removed" is against the neighbourhood median, which is the baseline named in the bar. The neighbourhood-by-room-type median is reported beside it. In Seattle that second baseline is a little stronger (78.5 against 79.5). The model still beats it. It does not beat either one by the 10% the bar asked for, and R² stays under 0.10 in every city. The model did not clear the bar. Seattle's 9.9% is not 10%, and its R² is 0.078.

Development R² for the model that was kept, mean ± standard deviation across seeds 0–4:

| City | R² | MAE |
| --- | ---: | ---: |
| Austin | 0.046 ± 0.036 | 66.22 ± 3.15 |
| San Diego | 0.049 ± 0.013 | 74.34 ± 2.28 |
| Seattle | 0.056 ± 0.019 | 71.79 ± 3.08 |

The neighbourhood median's development R² is negative in every city. A spatial split makes an area median a weak baseline, which is the same fact as in the price study. Beating a weak baseline by 9% is still a small R². Percentage error is a poor summary here: many labels are a handful of nights, so one 70-night miss is a huge percent. The median absolute percent error is the 52.9%, 45.3%, and 42.2% in the table.

A typical miss of about 70 nights, on a label whose median is 90, 124, or 144, is not a forecast a person should budget with.

### Low, medium, and high

The same features were also trained as a three-class model. The cuts are the training tertiles of the nights label, so the test labels do not set the boundaries. Low is at or under 48 nights in Austin, 72 in San Diego, and 96 in Seattle. High is over 132, 186, and 210. The baseline predicts the training majority class in that neighbourhood, and the city majority when the neighbourhood is new.

| City | Accuracy | Neighbourhood accuracy | Balanced accuracy | Baseline balanced | Cleared |
| --- | ---: | ---: | ---: | ---: | --- |
| Austin | 43.1% | 32.6% | 42.6% | 34.2% | yes |
| San Diego | 43.0% | 32.7% | 42.8% | 30.8% | yes |
| Seattle | 41.2% | 40.1% | 40.2% | 35.0% | no |

The class bar was also written down first: accuracy at least 8 points above the neighbourhood class, and a higher balanced accuracy. Austin and San Diego clear it. Seattle does not. The probabilities are not a calibration success. When the model says "medium," the average predicted probability is about 0.44 to 0.45, and the share that are actually medium is 36% in Austin, 35% in San Diego, and 30% in Seattle. The middle band is over-confident. No calibrator was fit after that look.

### Revenue

Estimated annual revenue is the predicted nightly rate times the predicted nights. The rate is the published price specification, refit so the seed-31 listings, their hosts, and anything within 1 km are out of that fit. The saved price artifact is not replaced, and the seed-23 price table is not a product of this refit. The target revenue is the advertised rate times the review-based nights.

| City | Revenue R² | Revenue MAE | R² if the nights were known | R² if the rate were known |
| --- | ---: | ---: | ---: | ---: |
| Austin | 0.133 | $12,906 | 0.681 | 0.264 |
| San Diego | -0.034 | $18,720 | 0.766 | -0.027 |
| Seattle | 0.117 | $10,470 | 0.739 | 0.323 |

Knowing the nights and predicting only the rate looks like the price model: R² 0.681, 0.766, and 0.739 on this split. Knowing the rate and predicting only the nights is much weaker, and in San Diego it is negative. The product inherits the nights error. San Diego's revenue R² of -0.034 means the product is worse than predicting the city's average estimated revenue.

### The grade

The grade is a rule, not a model. Inside each city, predicted annual revenue is compared with the training listings' estimated revenue (advertised rate times review-based nights). The cuts were fixed from those training rows. A value on a boundary takes the higher grade.

| Grade | Words | Rule |
| --- | --- | --- |
| A | Strong | at or above the 80th percentile |
| B | Above average | at or above the 60th, below the 80th |
| C | Typical | at or above the 40th, below the 60th |
| D | Below average | at or above the 20th, below the 40th |
| F | Weak | below the 20th percentile |

| City | 20th | 40th | 60th | 80th |
| --- | ---: | ---: | ---: | ---: |
| Austin | $3,240 | $7,560 | $14,400 | $25,182 |
| San Diego | $7,252.80 | $15,590.60 | $26,346 | $43,024.80 |
| Seattle | $6,402 | $13,020 | $19,929 | $29,104.20 |

On the holdout, the letter matched the grade of the review-based revenue 30.6% of the time in Austin, 30.5% in San Diego, and 25.0% in Seattle. Five letters, so a random letter is right about a fifth of the time. Thirty percent is not a rating. The app still shows the letter, next to the error, because a rank is readable. It is a rank of a noisy gross. It does not know the purchase price, the mortgage, taxes, insurance, cleaning, platform fees, or whether a short-term rental is legal at the address. If a purchase price is typed in, the app divides predicted revenue by that price and calls the result a rough gross yield. That is the division and nothing else.

### What this does not support

No resume line that says the model predicts how many nights a home will book is justified. The held-out R² is 0.073, 0.098, and 0.078, and the pre-set bar was missed in every city. No line that says the grade measures how good a rental the home would be is justified either. The letter is a within-city rank of a gross that, in San Diego, does not track the target.

The price result is a separate study and is not re-opened here.

## A calendar panel for a later label

The holdout above is a model of the review proxy. It is not evidence about real bookings, because the public calendar on one day does not say which unavailable nights were sold. The way to get a real label is to watch the same calendars until nights flip.

The collector takes a fixed panel from the newest Inside Airbnb listings for Austin (2026-09-17), San Diego (2026-09-20), and Seattle (2026-09-20). A listing is eligible when it has a review in the twelve months before that file and at least one open night. Seed 17, stratified by neighbourhood and room type. The ids are `data/calendar_sample.json`. The eligible rows are `data/calendar_pool/`. Each day the job stores `listing_id`, `date`, `available`, `min_nights`, `price` when present, and `fetched_at`, gzipped, on the `calendar-data` branch. The client waits about 2 seconds between listings, backs off on ordinary server errors, and stops when it is blocked or challenged. It does not log in, solve a captcha, or use a proxy. If the public calendar cannot be read from the runner, the same panel is cut out of Inside Airbnb's published calendar file and the run report says that. One published file has no flip to measure, so that path does not produce a booked-night rate.

On 2026-11-19 a probe asked 36 eligible ids per city, with the ids read as text. GitHub Actions has not executed the workflow. The cron starts from the default branch after merge, so whether a hosted runner can fetch is still open. The public endpoint answered on the machine that prepared the change: Austin 35 of 36, San Diego 35 of 36, Seattle 36 of 36. Failed was 0, and the probe was not challenged. The daily ask is 309, 309, and 300 so the expected number of calendars is about 300. That pass is under an hour at the probe's pace, inside the 180-minute job.

The same day, a first panel of 300 ids from the 2025 study files returned 80, 93, and 92 calendars. Those files are the 2026-11-19 commit on `calendar-data`. They are retired. About 90 calendars a city is too thin for the 26-week label, which is why the panel was redrawn. The label command keeps the current sample. Listing ids stay text: rounding an 18-digit id through a float asks for a different home, and that calendar comes back empty.

The inference, in `data/calendar_labels.py`, follows the daily-scrape rule that a booking is an available night that later becomes unavailable, plus the lead-time and length cuts that separate a short guest stay from a host block:

- Booked: a contiguous run of 1 to 13 nights that was still open on a snapshot inside 45 days of the check-in, then closed, and at least as long as the posted minimum stay.
- Owner block: 14 or more nights already closed when the check-in was 90 days out. Ninety days is the shortest availability window hosts can set. The same label is used when half or more of the nights that were open close between two snapshots at most 3 days apart.
- Closed window: 21 or more unavailable nights running through the last date the snapshot publishes, never seen open.
- Left out of the booked count: nights that were already unavailable on the first snapshot, nights that open again before the date, nights with no snapshot within 2 days, and runs whose lead time or length falls between the two cuts.

The rate is booked nights divided by booked nights plus nights still open at the date. Blocks and censored nights are counted beside it. A one-night owner hold can be called a booking when the minimum stay is 1. A fortnight stay, and a stay booked more than 45 days ahead, stay uncertain. Price is stored and unused, because the public calendar often leaves it blank on open nights and closed nights alike.

The first retrain waits 26 weeks. Nights already shut on the first day are censored. A night first seen beyond the 90-day line is a resolved outcome only once that date arrives, which is about 13 weeks after collection starts. A listing-level rate near 30% needs about 80 offered nights before its standard error is near 5 points (0.3 times 0.7, divided by 0.05 squared, is 84). That is about 12 further weeks. 13 + 12 is 25, and 26 weeks pulls the window off a single peak. The target at that point is the observed rate over the tracked window, not an annualized copy of it. An annual count, comparable to the review proxy, waits for 52 weeks. Occupancy accuracy will be re-tested on a fresh spatial holdout once the 26 weeks exist: a new seed, not seed 31 and not the price seeds, cells frozen before any fit, scored once. The R² figures in the table above are about the review proxy. They are not a forecast of that test.

## How to reproduce

From the repository root, with the pinned `requirements.txt`:

```bash
pip install -r requirements.txt
python -m evaluation.run_eval
python -m evaluation.run_occupancy
pytest
```

The price evaluation refits the development splits, writes the report, and scores seed 23 once. It then scores seed 7 and seed 11 once, as secondary comparisons. It does not reopen v3's feature search. v3's published metrics stay in `reports/v3_metrics.json`. The nights command writes `reports/occupancy_metrics.json` and scores seed 31 once. It does not replace the price artifacts.

The local app, after that:

```bash
streamlit run app/streamlit_app.py
```
