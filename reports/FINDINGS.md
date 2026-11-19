CapScout is a personal project built in spare time for learning. It is not a commercial product.

# What the nightly-rate model actually does

This is a study guide, not a product pitch. The question is the same one as the last round: given a public Airbnb listing in one city, how close can we get to the nightly price on that listing, without training on the house next door or on the same host?

The short answer, on the same locked spatial test as v2 (seed 7), scored once after the choices below were locked:

| City | Test R² | Test MAE | Test RMSE | MAPE | Median APE | 80% coverage | Test listings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin | 0.451 | $83.95 | $156.32 | 32.0% | 27.2% | 73.2% | 2,796 |
| San Diego | 0.675 | $99.52 | $178.94 | 29.8% | 22.5% | 77.1% | 875 |
| Seattle | 0.526 | $38.75 | $74.95 | 27.0% | 21.6% | 83.2% | 2,042 |

R² is the share of price variance the predictions capture. MAE is the average absolute miss, in dollars per night. RMSE punishes a few large misses more than MAE does. MAPE is the average of |miss| / price. Median APE is the same idea for a typical listing. Coverage is the share of test listings whose advertised price fell inside the 80% interval.

Those numbers are the result. The gain over v2 is real in Austin and small or absent in the other two cities. It is not a new model of the market.

The code is `python -m evaluation.run_eval`. The raw output is `reports/metrics.json`, `reports/locked_test_predictions.csv`, and `reports/confirmation_predictions.csv`. Sources are in `DATA.md`. The reading list for this round is `docs/RESEARCH.md`.

## Against v2, on the same rows

v2's published scores, from the archived `reports/v2_metrics.json`, and a refit of that exact specification in this run:

| City | Published v2 R² | Published v2 MAE | This round R² | This round MAE |
| --- | ---: | ---: | ---: | ---: |
| Austin | 0.422 | $87.74 | 0.451 | $83.95 |
| San Diego | 0.678 | $98.96 | 0.675 | $99.52 |
| Seattle | 0.532 | $38.19 | 0.526 | $38.75 |

The refit of the v2 specification, same rows, same fitting code, matched the published scores exactly. The difference in the table is the new columns, not a change in the trees' threading. Austin's MAE fell by $3.79. San Diego's rose by $0.56. Seattle's rose by $0.56. San Diego's median absolute percent error did improve, from 24.0% to 22.5%, and its MAPE from 30.4% to 29.8%. A few large misses got a little worse while a typical listing got a little closer.

On the same rows, the other baselines did not move, because they do not use the new columns. A neighbourhood-by-room-type median is still MAE $127.68, $187.07, and $61.90. The size-and-location specification is still R² 0.344 / 0.523 / 0.458 and MAE $93.21 / $117.45 / $43.07. Ridge on the new columns is R² 0.304 / 0.534 / 0.464 and MAE $94.80 / $112.18 / $39.00. In Seattle, ridge's MAE ($39.00) is close to the tree blend ($38.75). The protocol did not switch to ridge. Ridge is the linear baseline.

| City | Model | R² | MAE | RMSE | MAPE | Median APE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin | Neighbourhood × room | -0.172 | $127.68 | $228.43 | 53.1% | 42.7% |
| Austin | Previous specification | 0.344 | $93.21 | $170.88 | 38.5% | 31.6% |
| Austin | Ridge, same columns | 0.304 | $94.80 | $176.01 | 35.3% | 30.9% |
| Austin | v2 specification | 0.422 | $87.74 | $160.46 | 33.6% | 29.0% |
| Austin | This round | 0.451 | $83.95 | $156.32 | 32.0% | 27.2% |
| San Diego | Neighbourhood × room | 0.006 | $187.07 | $312.82 | 65.9% | 48.1% |
| San Diego | Previous specification | 0.523 | $117.45 | $216.64 | 33.8% | 27.1% |
| San Diego | Ridge, same columns | 0.534 | $112.18 | $214.05 | 30.7% | 25.8% |
| San Diego | v2 specification | 0.678 | $98.96 | $178.04 | 30.4% | 24.0% |
| San Diego | This round | 0.675 | $99.52 | $178.94 | 29.8% | 22.5% |
| Seattle | Neighbourhood × room | -0.053 | $61.90 | $111.69 | 46.7% | 34.5% |
| Seattle | Previous specification | 0.458 | $43.07 | $80.16 | 30.3% | 24.1% |
| Seattle | Ridge, same columns | 0.464 | $39.00 | $79.65 | 25.6% | 20.5% |
| Seattle | v2 specification | 0.532 | $38.19 | $74.44 | 26.6% | 21.1% |
| Seattle | This round | 0.526 | $38.75 | $74.95 | 27.0% | 21.6% |

## Why this test was not used to choose anything

Seed 7 was published with v2. Those scores had already been read, so using them to pick a feature would have been tuning on a test that was already open. Every choice below was made on development folds that do not include the seed-7 listings.

Because that test had been read, this round also froze a second spatial holdout before any new model was fit. Seed 11, written down and not searched, holds out 20% of the development cells. The ids are in `evaluation/confirm_ids.json`. They contain no prices. Those cells were inside v2's cross-validation, so the confirmation set is not a blind test of the v2 specification. It is a blind test of the choices made here. It was scored once, after the choices, and it was not used to change them.

| City | Confirmation listings | This round R² | This round MAE | v2 spec MAE | 80% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Austin | 1,389 | 0.483 | $62.36 | $64.42 | 84.4% |
| San Diego | 4,401 | 0.504 | $70.95 | $72.22 | 79.0% |
| Seattle | 216 | 0.578 | $45.29 | $45.85 | 81.5% |

On that check the new columns lowered MAE in every city, by about $2 in Austin, $1 in San Diego, and $0.50 in Seattle. Seattle's check is 216 listings. That is a small number, and the R² there went the other way (0.578 against 0.589 for the v2 specification). That split is too small to hang a claim on.

The locked-test model is trained on all of the development listings, including the confirmation cells, which is the same training population v2 used. The confirmation score uses a model that never saw those cells, after the same host purge and 1 km buffer.

## What was added, and what was not

v2's specification was not reopened. It is still size, room type, coordinates, distance to downtown and the airport, census tract income, rent, home value, and density, plus the listing's own amenities, reviews, and host fields. The model is still an equal-weight blend of CatBoost and XGBoost on log price, depth 4, no monotone constraints. LightGBM and the monotone constraint lost their bars in v2 and were not searched again.

On top of that, four add-ons were tried in a fixed order. Each one had to cut mean development MAE by at least 1% to stay. The folds are the five spatial seeds, on the cells in neither holdout. Mean MAE is an unweighted average of the city-seed scores, so a big city does not outvote a small one.

| Add-on, alone on the v2 columns | Mean development MAE | Kept? |
| --- | ---: | --- |
| v2 specification | $59.85 | the starting point |
| Beds, shared bath, host operations, review aspects, extra amenities | $58.62 | yes, this is the stack |
| Fixed phrases in the title and description | $60.12 | no |
| TF-IDF of the ad, eight components, fit inside the fold | $59.99 | no |
| Gaussian smooth of nearby training prices, 5 km | $60.04 | no |

The detail block was a 2.0% drop, from $59.85 to $58.62, which cleared the bar. The other three were worse than the v2 specification on their own. Tried on top of the detail block, they were worse than the detail block too (lexicon $59.15, wording $58.80, kernel $58.90). None of them stayed.

Per city, mean ± standard deviation of MAE on those same folds. "Selected" is the detail block. It matches the detail row because nothing was stacked on top of it.

| Set | Austin | San Diego | Seattle |
| --- | ---: | ---: | ---: |
| v2 specification | 66.66 ± 6.91 | 66.68 ± 15.97 | 46.20 ± 7.11 |
| Detail block | 64.47 ± 5.78 | 65.55 ± 15.53 | 45.86 ± 6.62 |
| Lexicon alone | 67.05 ± 6.37 | 66.64 ± 16.26 | 46.65 ± 6.40 |
| Wording alone | 66.57 ± 6.77 | 66.75 ± 16.16 | 46.64 ± 6.76 |
| Kernel alone | 66.98 ± 6.99 | 66.98 ± 16.09 | 46.15 ± 6.82 |
| Selected | 64.47 ± 5.78 | 65.55 ± 15.53 | 45.86 ± 6.62 |

The chart is `reports/figures/ablation_mae.png`. Read the differences. The detail block is worth about $2 in Austin, $1 in San Diego, and $0.30 in Seattle on these folds. Text and the kernel are flat or a bit worse. These folds are not v2's development folds. v2's published development MAEs included the confirmation cells, so they are not a before-and-after for this table.

Depth, on the kept columns, same folds:

| Depth | Mean MAE |
| --- | ---: |
| 4 | $58.62 |
| 6 | $59.34 |
| 8 | $60.18 |

Depth 4 is the best, not merely within 1% of the best. Deeper trees were worse, same as v2.

A single model for all three cities, with a city indicator, was $58.08 against $58.62 for the per-city blend. That is about 0.9%, short of the 1% bar, so the scored model stays per-city. Seattle was the reason to try it. Seattle's training set after the buffer is 1,507 listings. Pooling moved some Seattle folds and not enough of the average to pay for a model that is harder to explain.

Development results for the blend that was kept, mean ± standard deviation across the five seeds:

| City | R² | MAE |
| --- | ---: | ---: |
| Austin | 0.564 ± 0.082 | 64.47 ± 5.78 |
| San Diego | 0.620 ± 0.051 | 65.55 ± 15.53 |
| Seattle | 0.473 ± 0.183 | 45.86 ± 6.62 |

Seattle's development R² moves a lot from seed to seed. One split's R² is not a stable law, which is also what the locked test is.

## What the new columns actually are

All of them come from the same Inside Airbnb listing file. No new download.

- Beds, and whether the bathroom text says "shared".
- Host tenure as of the snapshot date, response rate, acceptance rate, response time on a 1–4 scale, and whether identity is verified. A blank stays missing. It is not filled with zero.
- The separate review scores for cleanliness, location, accuracy, check-in, and communication.
- A few more amenity flags: fireplace, dishwasher, self check-in, patio, BBQ, EV charger, pets, elevator.

`review_scores_value` was not eligible. Guests score "value" by comparing the stay to the price, so the column is partly a consequence of the label. Availability, estimated occupancy, and estimated revenue were not eligible either. Inside Airbnb's own notes say an unavailable night is not a booked night, and those estimates move with the price.

The text that was tried and dropped is the title, the description, and the neighbourhood blurb. Median description length is about 460 characters, so the field is not empty. A fixed list of phrases (view, luxury, renovated, new construction, quiet, walkable, downtown, water) and a fold-fit TF-IDF compressed to eight components both failed the bar. Eight was written down in advance. It was not a searched width. The kernel that failed is a Gaussian weight, bandwidth 5 km, on the 80 nearest training listings, with a row's own price removed. v2 had already rejected neighbourhood means and a 40-neighbour median. This was the continuous version of the same idea, and it did not travel across the buffer either.

## Prediction intervals

The interval in the app is an 80% band. v2 used one radius on the log residual, the same percentage width for every home. That band was well calibrated on development folds and then covered 66.7% of Austin's locked test. This round compared three methods on the development folds only:

| Method | Worst-city coverage | Worst-city coverage in the expensive fifth | Mean of city median widths |
| --- | ---: | ---: | ---: |
| Log residual, one radius | 79.9% | 69.7% | $124 |
| Log residual, width scaled with the predicted price | 79.9% | 77.4% | $124 |
| Conformalized quantile regression | 79.4% | 71.0% | $154 |

The rule, written down first, keeps the log band unless a challenger covers the expensive listings at least 2 percentage points better, without dropping the worst city, without getting more than 35% wider, and without leaving the point estimate outside the band. The scaled band did that. The quantile band was wider and its expensive-listing coverage was worse than the scaled band, so it stayed a diagnostic. The test was not used to pick.

On the locked test the scaled band covered 73.2% of Austin (median width $133), 77.1% of San Diego ($189), and 83.2% of Seattle ($103). The same test, scored once with the other two methods and not used to switch:

| City | Log band | Scaled band (the one in the app) | Conformalized quantiles |
| --- | ---: | ---: | ---: |
| Austin | 71.7% | 73.2% | 76.7% |
| San Diego | 74.7% | 77.1% | 75.2% |
| Seattle | 83.5% | 83.2% | 83.7% |

Austin is still short of 80%. The expensive eighth of Austin predictions, roughly $274 to $1,129, was covered 80.3% of the time. The hole is no longer only the top of the market. A band around $84 to $100 was covered 61.6%. San Diego has a similar dip around $122 to $156 (63.3%). Seattle's cheapest bin is the weak one (75.8%), and the rest sit near the target. The chart is `reports/figures/calibration.png`.

Development coverage of the scaled band was 79.9% in Austin, 80.6% in San Diego, and 80.4% in Seattle. The test did not match that, which is what a spatial holdout is allowed to do. A held-out part of the city is not an exchangeable draw from the training listings, so the conformal guarantee does not fully apply. The radius was not widened after seeing 73.2%.

## What the model is using

SHAP values are the average absolute contribution to the log-price prediction, on a sample of up to 600 locked-test rows, averaged across the two models. The eight wording components would have been summed into one row. They are not in the model, so they are not in the chart. The new columns that do show up are small next to size and location. The chart is `reports/figures/shap.png`.

- **Austin:** bedrooms (0.22), bathrooms (0.10), reviews per month (0.09), capacity (0.09), distance to downtown (0.08). The first new column in the list is a BBQ flag (0.03), then host acceptance rate (0.03).
- **San Diego:** capacity (0.24) and longitude (0.17) still lead. Longitude is the coast. Then bathrooms and bedrooms. A shared bath is 0.03. The location review score is 0.02.
- **Seattle:** capacity (0.13), bedrooms (0.12), distance to downtown (0.06). Beds and a shared bath are 0.03 each. The location score is 0.03.

The detail block helps a little in total and no single new column dominates. That matches a $1 to $4 MAE change better than a story about one clever feature.

Partial dependence on development rows, holding the other columns fixed (`reports/figures/partial_dependence.png`):

- Bedrooms still raise the rate. Austin goes from about $118 for a studio to about $233 for five bedrooms. Seattle goes from about $120 to about $188. San Diego goes from about $181 to about $254.
- Distance to downtown still matters in Seattle (about $184 at 1 km, about $134 at 10 km) and barely moves San Diego. The coast, not the downtown pin, is San Diego's location signal.
- Tract median income moves Austin by about $9 across the middle of the distribution and does not draw a clean slope in Seattle.

## Where the errors are

Dollar error still grows with the predicted price. Austin's top fifth of predictions, roughly $212 to $1,129, has MAE about $171, and the model is still low there (mean residual about +$98). San Diego's top fifth, roughly $446 to $1,476, has MAE about $232. Seattle's top fifth has MAE about $80, and the bias is small.

Room type on the same test:

- Entire homes are most of the rows. Austin 2,417 homes, R² 0.453, MAE $86.17. San Diego 670 homes, R² 0.674, MAE $103.61. Seattle 1,685 homes, R² 0.487, MAE $41.66.
- Private rooms are cheaper, so the dollar miss is smaller. Austin MAE $64.86 and R² 0.160. San Diego MAE $55.34 and R² 0.388. Seattle MAE $25.03 and R² 0.434.
- Hotel rooms are rare and the model misses them by a couple of hundred dollars. Do not quote a hotel-room R².
- Shared rooms are a handful of listings, including one San Diego row. An R² on that is not a result.

## The old R² near 0.97

The previous training target, in a pipeline this study does not use, was not the listing's price. Listings were clustered geographically, grouped by cluster and bedroom count, and each row's target was replaced by that group's median price. The model was then scored against those medians.

On Austin, as a demonstration and not as a result (it uses the whole city, including what would have been a test):

- A model trained on the group median and scored against that median has cross-validated R² **0.973** and MAE about **$7**.
- The same predictions, scored against the listing's own price, have R² **0.385** and MAE **$84.40**.

That is the inflation. This study does not train on a median.

## What we tried that did not help

**Listing text.** The descriptions are there. A fixed lexicon and an eight-component TF-IDF, vocabulary fit on the training fold only, both failed the 1% bar, alone and on top of the detail block. A lot of what the ad says is size, a room type, and an amenity the structured columns already have. What is left did not survive a spatial split. A neural embedding of the same text was not fit. It would have been another model download, harder to explain in an interview, and the linear compression of the same words had already failed. Photos were not downloaded. The picture URL points at a host's image. That image is not part of the CC BY listing table.

**Nearby prices, again.** A 5 km Gaussian smooth of training log prices, leaving out the row itself, was a bit worse than the v2 specification. v2's neighbourhood means were actively harmful, and v2's 40-neighbour median missed the 1% bar. Three versions of "what do the places over there charge" have now failed under this buffer. The buffer is doing what it was built to do.

**One model for three cities.** About 0.9% better on the development folds. The bar was 1%. Not kept.

**Deeper trees.** Worse at depth 6 and worse again at depth 8.

**Conformalized quantile regression, as the interval.** Better than the old log band on Austin's test (76.7% against 71.7% for the log band of this same model), and wider by about $30. On the development folds, which are the only place the choice was allowed to look, it did not cover the expensive listings as well as the scaled band, and it was wider. It is in the table above so the test result is visible. It is not the band in the app.

## Every cleaning step

Cleaning is in `data/ingest.py`. The rules are fixed dollar amounts, not percentiles of the file. A percentile computed before the split would let the test listings help decide who counts as an outlier.

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

**One locked test.** Seed 7 holds out 20% of cells. The ids are in `evaluation/locked_test_ids.json`. They were written from coordinates and the seed. The code refuses to regenerate them if they no longer match.

| City | Test listings | Share of rows | Training rows before the blocks | Removed for a shared host | Removed by the 1 km buffer | Training rows used |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin | 2,796 | 27.2% | 7,500 | 1,501 | 1,602 | 4,397 |
| San Diego | 875 | 7.5% | 10,715 | 1,610 | 627 | 8,478 |
| Seattle | 2,042 | 34.1% | 3,950 | 1,015 | 1,428 | 1,507 |

**The confirmation holdout** is 20% of the remaining cells, seed 11. Austin 1,389 listings, San Diego 4,401, Seattle 216. San Diego's confirmation cells happened to be the dense ones, so the selection pool there is 4,077 listings rather than the full 8,478. That is the split working. It also means San Diego's development MAE is noisier than a reader might think from the city's size.

**Tuning never sees either holdout.** The selection folds are five spatial splits, seeds 0 through 4, of the pool that is in neither holdout. Early stopping is a further spatial slice inside the training side of whatever split is being fit.

**The target is log(1 + price).** Every reported metric is computed after converting predictions back to dollars.

## Self-audit

Checked in code and recorded in `reports/metrics.json`:

- The frozen listing ids match seed 7 recomputed from coordinates. The confirmation ids match seed 11, and the two lists do not overlap. Neither file contains prices.
- No spatial cell is in both training and test, for the locked test or the confirmation split.
- No host is in both sides. No training listing is within 1 km of a held-out listing.
- The detail columns do not include a price, a value score, availability, or an estimated revenue.
- Feature add-ons, depth, pooling, and the interval were chosen by the functions in `evaluation/select.py` from development MAE or development coverage only. `tests/test_report.py` replays those functions on the JSON and fails if the recorded winner was edited. The same test replays v2's archived choices.
- The locked-test metrics in the JSON match `reports/locked_test_predictions.csv`. The confirmation metrics match `reports/confirmation_predictions.csv`.

`audit_passed` in the JSON is true.

## What the local app is doing

`streamlit run app/streamlit_app.py` serves `127.0.0.1` only. It is a way to read one prediction and the test report. It is not a second evaluation and it is not deployed.

- The nightly rate is the blend.
- The band is the scaled log interval from this study.
- The reasons are SHAP values on that row, turned into a sentence and an approximate percent.
- Nearby rates are Inside Airbnb advertised prices. They are not inputs to the model. The page says whether the estimate sits above or below the median of those five.
- An address is geocoded with the Census geocoder. The map can also take a click.
- Description, amenities, reviews, and superhost status are optional. Left blank, they stay missing. Missing is not "no kitchen."

## Limitations

- The label is the advertised rate, not what a guest paid and not what a host cleared after fees.
- The improvement over v2 is a few dollars in Austin and noise in San Diego and Seattle. That is the result, not a rounding choice.
- Austin's 80% interval covered 73.2% of the locked test. It is better than 66.7% and it is not 80%.
- Airbnb jitters coordinates by up to about 150 meters. Distances, cells, and the tract join are fuzzy at that scale.
- The 1 km buffer and the host purge throw away a large share of training rows. Seattle keeps 1,507 training listings against a test of 2,042.
- San Diego's locked test is 875 listings. Seattle's confirmation check is 216. One unlucky set of cells moves those numbers.
- Hotel rooms and shared rooms are too rare to judge.
- Expensive listings are still under-predicted.
- Development seed-to-seed MAE in San Diego still moves by about $16. One split's MAE is not a stable law.
- The saved model has not seen the locked-test listings. That is intentional. It has seen the confirmation listings, because those are part of the training population for the locked-test model. The confirmation score itself uses a different fit that has not seen them.
- Three cities, one protocol. Nothing here is a national model.
- Text and photos might still carry a signal this round did not extract. The round measured one lexicon, one eight-component TF-IDF, and no images. It did not measure every way to read an ad.

## How to reproduce

From the repository root, with the pinned `requirements.txt`:

```bash
pip install -r requirements.txt
python -m evaluation.run_eval
pytest
```

The evaluation refits the development splits, writes the report, and scores the locked test and the confirmation holdout once. It does not reopen v2's feature search. v2's published metrics stay in `reports/v2_metrics.json`.

The local app, after that:

```bash
streamlit run app/streamlit_app.py
```
