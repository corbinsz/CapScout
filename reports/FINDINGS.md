CapScout is a personal project built in spare time for learning. It is not a commercial product.

# What the nightly-rate model actually does

This is a study guide, not a product pitch. The question is: given a public Airbnb listing in one city, how close can we get to the nightly price on that listing, without training on the house next door or on the same host?

The short answer, on a test set that was frozen before the new features were built and scored once at the end:

| City | Test R² | Test MAE | Test RMSE | MAPE | Median APE | Test listings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin | 0.422 | $87.74 | $160.46 | 33.6% | 29.0% | 2,796 |
| San Diego | 0.678 | $98.96 | $178.04 | 30.4% | 24.0% | 875 |
| Seattle | 0.532 | $38.19 | $74.44 | 26.6% | 21.1% | 2,042 |

R² is the share of price variance the predictions capture. Zero means predicting the average price would have done as well. MAE is the average absolute miss, in dollars per night. RMSE punishes a few large misses more than MAE does. MAPE is the average of |miss| / price, in percent. Median APE is the same idea for a typical listing, so one luxury miss does not set it.

Those numbers are the result. Everything below is what they mean, how they were produced, and what not to claim.

The code is `python -m evaluation.run_eval`. The raw output is `reports/metrics.json` and `reports/locked_test_predictions.csv`. The figures are in `reports/figures/`. Sources and the join methods are in `DATA.md`.

## In plain English

We kept the three cities from the previous round, Austin, San Diego, and Seattle, because the other Inside Airbnb snapshots on disk are either several markets in one file or missing a price or a bedroom count on too many rows. We froze a new spatial holdout, seed 7, and wrote the held-out listing ids down before any census or map feature was scored. The previous round had already published a seed-42 test. Reusing it would have meant tuning against a test that had been read.

On the development cells we compared, in an order written down first: size, then location, then census tract attributes, then OpenStreetMap, then building type. A simpler set won if it was within 1% mean absolute error of the best set. Census won that comparison. Map features and building type were close and not better enough to add. Two ways of using other listings' prices, both fit inside the training fold only, did not help. The listing's own amenities and reviews did: development MAE fell from about $64 to about $59, which cleared a pre-set 1% bar, including in Seattle.

The point model is a blend. CatBoost was a little better than XGBoost on the development splits and not 2% better, so it did not replace XGBoost. An equal-weight average of the two was just over 1% better than XGBoost alone, which was the bar for keeping a blend. Monotone constraints, which would have forced more bedrooms to never lower the prediction, were slightly worse and were dropped. The trees learned that direction anyway.

On the locked test the blend beats a neighbourhood median and beats the previous specification (size, location, XGBoost, depth 4) in every city. It does not beat them by the same amount, and San Diego's test is only 875 listings, so that city's R² of 0.678 is one split, not a stable law. The 80% intervals, calibrated on development folds to cover about 80%, covered 66.7% of Austin's test, 75.2% of San Diego's, and 86.6% of Seattle's. The spatial shift the split was built to create is also a shift in the errors. The interval is a range, not a guarantee.

## Where the data came from

Listings are Inside Airbnb snapshots, CC BY 4.0. Credit Inside Airbnb. Dates and the cleaning rules are below. We did not scrape Airbnb.com.

New columns, measured on the development splits:

- **Census ACS 2024 5-year**, public domain, joined by point-in-polygon to TIGER/Line 2024 tracts. Median household income, median gross rent, median home value, population density. About 99.8% of rows landed in a tract polygon. A missing estimate is the city median of tracts, not a function of listing prices.
- **OpenStreetMap**, ODbL, © OpenStreetMap contributors, via Overpass. Distance to the nearest food place, transit stop, park, beach, and attraction, and counts within 1 km. Built, measured, and not used in the scored model.
- **The listing's own ad:** amenities, minimum nights, review score and volume, superhost, host portfolio size, instant book. These are known for an existing ad. They are not known for a house that has never been listed. The local app leaves them missing in that case.

The calendar was not used. Inside Airbnb says an unavailable night is not necessarily a booked night, and availability depends on the price, so it would be a leak.

A previous study of a different split, with size and location only, reported:

| City | That study's R² | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Austin | 0.426 | $95.61 | $164.78 |
| San Diego | 0.555 | $85.02 | $170.79 |
| Seattle | 0.437 | $49.80 | $95.16 |

Do not put those next to the table at the top and call it an improvement. The rows are different. The same-rows comparison is later, against a model with that older specification trained on this study's development listings.

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

## The features that were eligible

Stage 1, each one added on top of the one above, compared at XGBoost depth 6 so depth does not get mixed into the feature decision:

1. **Size and room type.** Bedrooms, bathrooms, capacity, and flags for a private room, a shared room, and a hotel room. An entire home is the baseline, all three flags off.
2. **Location.** Latitude, longitude, distance to downtown, distance to the airport. Coordinates let a tree draw a surface. The two distances are the part a person can say out loud.
3. **Census.** Tract income, rent, home value, density. Slow-moving, and not a function of the advertised rate.
4. **OpenStreetMap.** Distances and 1 km counts. A cap of 20 km keeps "no beach in Austin" from becoming a huge number.
5. **Structure.** House, cabin, cottage, condo, townhouse, guest unit, and two interactions: capacity times a private or shared room, and downtown distance times house or condo. An extra guest does not price like a whole home when the product is a room.

Then, on whatever stage 1 kept, two encodings that use other listings' prices, fit on the training fold only:

- A smoothed neighbourhood mean of log price, and the same thing by bedroom bin. Unseen neighbourhoods fall back to the training mean. The prior is 20 listings, so a block of three sales cannot memorize itself.
- The median log price of the 40 nearest other training listings. Under a 1 km buffer those neighbours are not next door.

An encoding had to improve mean development MAE by at least 1%. If both did, the smoothed one would win a tie, because it is the more stable of the two. Neither cleared the bar.

Then listing quality, same 1% bar. It cleared it.

## How the data was split

**A cell, not a row.** Each listing is assigned to a grid cell of 0.03 degrees, about 3 km of latitude, from its coordinates alone.

**One locked test.** Seed 7 holds out 20% of cells. The fraction is of cells, not of rows, so the row share is not 20%. The ids are in `evaluation/locked_test_ids.json`. They were written from coordinates and the seed, and the code refuses to regenerate them if they no longer match.

| City | Test listings | Share of rows | Training rows before the blocks | Removed for a shared host | Removed by the 1 km buffer | Training rows used |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin | 2,796 | 27.2% | 7,500 | 1,501 | 1,602 | 4,397 |
| San Diego | 875 | 7.5% | 10,715 | 1,610 | 627 | 8,478 |
| Seattle | 2,042 | 34.1% | 3,950 | 1,015 | 1,428 | 1,507 |

San Diego's test is a small set of cells. Seattle's test is a large set, and the buffer then removes most of what would have been the training side. Both facts are the split working, and both make a single MAE less stable than it looks.

**Tuning never sees the test.** The development rows are split the same way five times, seeds 0 through 4. Mean MAE across those seeds, averaged over the three cities, makes every modeling choice. The spread we report is the sample standard deviation across the five seeds. It is a spread, not a confidence interval from a formula.

**Early stopping is inside the training side** of whatever split is being fit. A further 20% of those training cells pick the number of trees. That slice is not the validation fold and not the locked test. The model that scores the locked test is then refit on all of the development rows, with that tree count.

**The target is log(1 + price).** A 20% miss on a $100 night is treated like a 20% miss on a $400 night. Every reported metric is computed after converting predictions back to dollars. We do not report a log-space R² and call it the result.

## What was selected

Development MAE, averaged across the three cities and five seeds. Stage 1 is XGBoost depth 6. The best stage-1 MAE was $63.84, for structure. Census was $64.29, which is within 1%, and it is the simpler set, so census is what we kept. Location alone was $65.26, which is outside that band.

| Feature set | Mean development MAE | In the model? |
| --- | ---: | --- |
| Size and room type | $68.93 | no, location was better by more than 1% |
| Plus location | $65.26 | no |
| Plus census | $64.29 | yes, this is the stage-1 set |
| Plus OpenStreetMap | $64.45 | no |
| Plus structure | $63.84 | no, within 1% of census, and census is simpler |
| Location plus OpenStreetMap, no census | $65.15 | not eligible, a substitution check |
| Census plus smoothed neighbourhood price | $70.88 | no, worse |
| Census plus 40 nearest training prices | $64.15 | no, short of a 1% gain |
| Census plus listing quality | $58.70 | yes |

Per city, the same ablation, mean ± standard deviation of MAE. This is the table that says what each source did, rather than the average hiding a city.

| Set | Austin | San Diego | Seattle |
| --- | ---: | ---: | ---: |
| Size and room | 83.63 ± 15.06 | 71.50 ± 13.87 | 51.67 ± 3.82 |
| Plus location | 81.79 ± 14.73 | 64.04 ± 17.87 | 49.95 ± 5.99 |
| Plus census | 79.96 ± 16.37 | 64.01 ± 16.89 | 48.91 ± 6.59 |
| Plus OpenStreetMap | 80.18 ± 15.29 | 63.28 ± 17.30 | 49.89 ± 5.86 |
| Plus structure | 79.56 ± 14.87 | 63.38 ± 16.82 | 48.58 ± 6.10 |
| OSM without census | 81.41 ± 14.15 | 64.10 ± 18.13 | 49.93 ± 6.04 |
| Smoothed neighbourhood price | 85.36 ± 13.68 | 68.92 ± 18.52 | 58.36 ± 6.06 |
| 40 nearest training prices | 79.34 ± 15.16 | 63.48 ± 17.05 | 49.62 ± 6.24 |
| Listing quality | 71.21 ± 16.15 | 59.13 ± 14.15 | 45.77 ± 5.73 |

Read it as differences, not as a leaderboard. Location is worth a few dollars in every city, and more than that in San Diego. Census is worth about $2 in Austin, about $1 in Seattle, and nothing visible in San Diego. OpenStreetMap on top of census is flat or a bit worse. Listing quality is the large, consistent gain: about $9 in Austin, $5 in San Diego, $3 in Seattle. The smoothed neighbourhood price is actively harmful, especially in Seattle, which is what a spatial split should do to a feature that only knows neighbourhoods the training cells happened to contain.

The chart is `reports/figures/ablation_mae.png`.

Depth, on the selected columns, same average:

| Depth | Mean MAE |
| --- | ---: |
| 4 | $58.37 |
| 6 | $58.70 |
| 8 | $59.22 |

Depth 4 is the best, not merely within 1% of the best. Deeper trees were worse.

Libraries at depth 4:

| Library | Mean MAE |
| --- | ---: |
| XGBoost | $58.37 |
| LightGBM | $58.83 |
| CatBoost | $57.70 |
| XGBoost with monotone constraints | $59.00 |
| Equal-weight CatBoost and XGBoost | $57.62 |

CatBoost is about 1.1% better than XGBoost. The bar for replacing XGBoost was 2%, so the single model stays XGBoost. The blend is about 1.3% better than XGBoost, and the bar for a blend was 1%, so the blend is the scored model. The weights were not fit on the validation fold. A learned stacker would have needed another split, and with three cities it would mostly be noise. Monotone constraints were about 1.1% worse. More bedrooms does raise the prediction, which the partial-dependence plots show, but forcing the constraint did not reduce dollar error.

Development results for the blend that was kept, mean ± standard deviation across five seeds:

| City | R² | MAE |
| --- | ---: | ---: |
| Austin | 0.562 ± 0.054 | 69.71 ± 15.11 |
| San Diego | 0.568 ± 0.070 | 58.28 ± 15.32 |
| Seattle | 0.544 ± 0.084 | 44.88 ± 5.48 |

Ridge on the same columns, same splits: Austin MAE 71.19 ± 15.01 (R² 0.554 ± 0.096), San Diego 62.34 ± 15.89 (R² 0.529 ± 0.025), Seattle 45.26 ± 5.72 (R² 0.582 ± 0.058). In Seattle, on the development splits, ridge's R² is higher than the blend's. We did not switch to ridge. The protocol's selected model is the tree blend, and ridge is the linear baseline. The locked test, below, prefers the blend in Seattle as well. Both facts stay.

The neighbourhood median's development R² is about zero or negative. The test cells are different parts of the city. A neighbourhood the training cells never saw falls back to a citywide median, and a citywide median is a bad guess for a held-out neighbourhood. That is what a spatial split is for.

## The locked test

Scored once, after the choices above.

| City | Model | R² | MAE | RMSE | MAPE | Median APE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Austin | Neighbourhood × room | -0.172 | $127.68 | $228.43 | 53.1% | 42.7% |
| Austin | × bedroom bin | 0.099 | $113.94 | $200.31 | 48.0% | 37.4% |
| Austin | Previous specification | 0.344 | $93.21 | $170.88 | 38.5% | 31.6% |
| Austin | Ridge | 0.258 | $99.60 | $181.72 | 37.2% | 34.2% |
| Austin | Blend | 0.422 | $87.74 | $160.46 | 33.6% | 29.0% |
| San Diego | Neighbourhood × room | 0.006 | $187.07 | $312.82 | 65.9% | 48.1% |
| San Diego | × bedroom bin | 0.327 | $156.69 | $257.29 | 59.6% | 40.2% |
| San Diego | Previous specification | 0.523 | $117.45 | $216.64 | 33.8% | 27.1% |
| San Diego | Ridge | 0.519 | $113.67 | $217.70 | 31.3% | 26.3% |
| San Diego | Blend | 0.678 | $98.96 | $178.04 | 30.4% | 24.0% |
| Seattle | Neighbourhood × room | -0.053 | $61.90 | $111.69 | 46.7% | 34.5% |
| Seattle | × bedroom bin | -0.053 | $61.94 | $111.67 | 46.7% | 34.5% |
| Seattle | Previous specification | 0.458 | $43.07 | $80.16 | 30.3% | 24.1% |
| Seattle | Ridge | 0.464 | $40.68 | $79.70 | 28.6% | 21.9% |
| Seattle | Blend | 0.532 | $38.19 | $74.44 | 26.6% | 21.1% |

The previous specification is XGBoost depth 4 on size, room type, and location only. Same development rows, same test rows, same fitting procedure. It is the fair "did the new columns help" comparison. They did: MAE falls by about $5 in Austin, $18 in San Diego, and $5 in Seattle. That bundle is census plus listing quality plus the blend, not census alone. The ablation is where those pieces are separated, and it was not re-run on the test.

Predicted versus actual, and residual versus predicted, are `reports/figures/pred_vs_actual.png` and `reports/figures/residuals.png`.

### Where the errors are

Dollar error grows with the predicted price. The top fifth of Austin predictions, roughly $204 to $1,079, has MAE about $167, and the model is still low there (mean residual about +$85). San Diego's pattern is the same shape. Seattle's dollar errors are smaller because Seattle's prices are smaller. Expensive listings are systematically under-predicted. A log target does that: it cares about relative error, and it shrinks extremes toward the middle.

Room type on the same test:

- Entire homes are most of the rows and carry the R² above. Austin 2,417 homes, R² 0.425, MAE $90.02. San Diego 670 homes, R² 0.669, MAE $103.90. Seattle 1,685 homes, R² 0.495, MAE $40.91.
- Private rooms are cheaper. Dollar MAE is smaller (Austin $68.00, San Diego $58.27, Seattle $25.33). Austin's private-room R² is only 0.101.
- Hotel rooms are rare and the model misses them by about $250. Do not quote a hotel-room R².
- Shared rooms are a handful of listings, including a single San Diego row. An R² on that is not a result.

## Prediction intervals

The interval in the app is an 80% band from split conformal on the log residual. For each development fold we took the absolute error in log price, held that fold out, and checked coverage with a radius fit on the other folds. That development coverage was 79.7% in Austin, 80.1% in San Diego, and 79.7% in Seattle. The radius applied to the test is fit on all of those development residuals, not on the test.

On the locked test the same radius covered 66.7% of Austin (median width $119), 75.2% of San Diego (median width $185), and 86.6% of Seattle (median width $109). The chart is `reports/figures/calibration.png`. Austin's band is too narrow, especially on the listings the model already thinks are expensive, which are the ones it under-predicts. Seattle's band is a bit wide. A spatial holdout is not an exchangeable sample, so the conformal coverage guarantee does not fully apply. We did not widen Austin's band after seeing 66.7%. That would be a second look at the test.

A quantile model, pinball loss at the 10th and 90th percentiles, fit on the development rows and scored once, covered 70.2%, 70.9%, and 71.8%. The two quantiles did not cross. Coverage is more similar across cities than the conformal band, and still short of 80%. It is the diagnostic, not the interval the app shows.

## What the model is using

SHAP values are the average absolute contribution to the log-price prediction, on a sample of up to 600 locked-test rows, averaged across the two models in the blend. They explain the model we scored. They were not used to drop features. The chart is `reports/figures/shap.png`.

- **Austin:** bedrooms (0.21), reviews per month (0.13), the host's listing count (0.11), bathrooms (0.11), capacity (0.09), private-room status (0.08), tract median home value (0.08), distance to downtown (0.07).
- **San Diego:** capacity (0.25) and longitude (0.19) lead. Longitude is the coast. Then bedrooms, bathrooms, and reviews per month. Census is not in the top eight. That matches the ablation, where census did not move San Diego's development MAE.
- **Seattle:** bedrooms (0.15), capacity (0.13), reviews per month (0.08), private-room status (0.07), distance to downtown (0.06), tract home value (0.05).

Partial dependence, on development rows, holding the other columns fixed (`reports/figures/partial_dependence.png`):

- Bedrooms raise the predicted rate in every city. In Austin the curve goes from about $119 for a studio to about $229 for five bedrooms. Seattle goes from about $113 to about $198. The monotone constraint was not needed for the trees to learn this.
- Distance to downtown lowers the rate, sharply in Seattle (about $183 at 1 km, about $134 at 10 km) and mildly in Austin. In San Diego the curve is almost flat. The coast, not the downtown pin, is the location signal, which is why longitude leads the San Diego SHAP chart.
- Tract median income moves Austin and San Diego by about $10 across the middle of the distribution, and it does not move Seattle in any clean direction. Home value shows up in SHAP for Austin and Seattle. Income is a real column and a small one.

## The old R² near 0.97

The previous training target, in a pipeline this study does not use, was not the listing's price. Listings were clustered geographically, grouped by cluster and bedroom count, and each row's target was replaced by that group's median price. The model was then scored against those medians.

On Austin, as a demonstration and not as a result (it uses the whole city, including what would have been a test):

- A model trained on the group median and scored against that median has cross-validated R² **0.973** and MAE about **$7**.
- The same predictions, scored against the listing's own price, have R² **0.385** and MAE **$84.40**.

That is the inflation. This study does not train on a median. The baselines use medians as predictions. The model's target is the listing.

## What we tried that did not help

**OpenStreetMap.** On top of census, mean development MAE went from $64.29 to $64.45. Without census, against location alone, it went from $65.26 to $65.15. Per city the moves are a dollar or less, and in Seattle adding the map features to census made MAE worse ($48.91 to $49.89). Parks are everywhere in these extracts, so "distance to a park" has little contrast. Beaches matter in principle in San Diego and not in Austin, and a single column for all three cities did not pay for itself. The features are still built by `python -m data.external`, so the negative result is reproducible rather than a missing download.

**Building type and the two interactions.** The best stage-1 number, and within 1% of census. Not worth the extra columns under the pre-set rule.

**Smoothed neighbourhood prices.** Worse in every city, and badly worse in Seattle ($48.91 to $58.36). A neighbourhood the training cells never saw falls back to a broad mean, and that constant is noise to a tree that already has coordinates and a tract income.

**Forty nearest training listings.** Slightly better than census alone in the average ($64.15 versus $64.29), short of 1%. The neighbours are at least 1 km away. They are not comps.

**Monotone constraints.** About 1% worse. The partial-dependence plot already slopes the right way for bedrooms.

**CatBoost as the only model.** About 1% better than XGBoost, short of the 2% bar. It is in the blend because the average cleared a different, pre-set bar, not because we liked the name.

**A learned stacker, a pooled three-city model, and the calendar.** Not fit. A stacker needs a further split or it flatters itself. Pooling would have been a second study. The calendar would have been a leak.

## Self-audit

Checked in code and recorded in `reports/metrics.json` under each city:

- The frozen listing ids match seed 7 recomputed from coordinates. The file does not contain prices.
- No spatial cell is in both training and test.
- No host is in both training and test.
- No training listing is within 1 km of a test listing. The minimum gap is just over 1 km in each city.
- The only columns whose names involve price would be the fold-fit encodings. Those encodings were not selected, and none of them are in the scored matrix.
- Census and OpenStreetMap columns are computed from coordinates and public maps. Shuffling the price column cannot change them. The tests cover the distance function and the tract join on a synthetic polygon.
- Feature sets, depth, library, the monotone decision, and the blend were chosen by the functions in `evaluation/select.py` from development MAE only. `tests/test_report.py` replays those functions on the JSON and fails if the recorded winner was edited.
- The locked-test metrics in the JSON match `reports/locked_test_predictions.csv` recomputed from the saved predictions, including the interval coverage.

`audit_passed` in the JSON is true.

## What the local app is doing

`streamlit run app/streamlit_app.py` serves `127.0.0.1` only. It is a way to read one prediction and the test report. It is not a second evaluation and it is not deployed.

- The nightly rate is the blend.
- The band is the conformal interval from this study, with the city's radius.
- The reasons are SHAP values on that row, averaged across the two models, turned into an approximate percent with exp(shap) − 1.
- Nearby rates are Inside Airbnb advertised prices for similar listings. They are not inputs to the model.
- An address is geocoded with the Census geocoder, which is public domain. The map can also take a click.
- Reviews and amenities are left missing unless you have them. Missing is not "no kitchen." The test metrics are for listings where those fields were observed. A never-listed home is a rougher use of the same model.

## Limitations

- The label is the advertised rate, not what a guest paid and not what a host cleared after fees.
- Airbnb jitters coordinates by up to about 150 meters. Distances, cells, and the tract join are fuzzy at that scale. Tract polygons in the cache are simplified by about 15 meters on top of that.
- The 1 km buffer and the host purge throw away a large share of training rows. Seattle keeps 1,507 training listings against a test of 2,042.
- Test size differs by city because cells are not equal. San Diego's test is 7.5% of rows. One unlucky set of cells moves that R².
- Hotel rooms and shared rooms are too rare to judge.
- Errors of $40 to $100 a night, and intervals $100 to $180 wide, are large relative to the price. The model is a level and a ranking, not a quote.
- Expensive listings are under-predicted, and Austin's 80% interval only covered 67% of the test.
- Development seed-to-seed MAE moves by about $15 in Austin and San Diego. One split's MAE is not a stable law.
- The saved model has not seen the test listings. That is intentional.
- Three cities, one protocol. Nothing here is a national model.
- Histogram trees can move the third decimal of a metric when every CPU core is used. The tables match `reports/metrics.json` from this run. That file is the record.

## How to reproduce

From the repository root, with the pinned `requirements.txt`:

```bash
pip install -r requirements.txt
python -m evaluation.run_eval
pytest
```

The evaluation refits every development split, then scores the locked test once. It writes `reports/metrics.json`, `reports/locked_test_predictions.csv`, the figures, and `models/artifacts/<city>/model.joblib`.

The local app, after that:

```bash
streamlit run app/streamlit_app.py
```
