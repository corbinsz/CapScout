CapScout is a personal project built in spare time for fun and learning. It is not a commercial product.

# What the nightly-rate model actually does

This is a study guide for the model, not a product pitch. The question is: given a public Airbnb listing in one city, how close can we get to the nightly price on that listing, without cheating?

The short answer, on a test set that was scored once:

| City | Test R² | Test MAE | Test RMSE | Test listings |
| --- | ---: | ---: | ---: | ---: |
| Austin | 0.426 | $95.61 | $164.78 | 2,087 |
| San Diego | 0.555 | $85.02 | $170.79 | 3,142 |
| Seattle | 0.437 | $49.80 | $95.16 | 1,012 |

R² is the share of price variance the predictions capture. Zero means “you would have done as well by predicting the average price.” MAE is the average absolute miss, in dollars per night. RMSE is the same idea but it punishes a few large misses more than MAE does.

Those three numbers are the result. Everything below is what they mean, how they were produced, and what not to claim.

The code that produced them is `python -m evaluation.run_eval`. The raw output is `reports/metrics.json` and `reports/locked_test_predictions.csv`. The figures are in `reports/figures/`.

## Where the data came from

The listings are Inside Airbnb snapshots, one file per city, under `data/cities.py` dates:

- Austin, scraped date 2025-06-13
- San Diego, 2025-03-16
- Seattle, 2025-03-16

Inside Airbnb’s public downloads are under the Creative Commons Attribution 4.0 license (CC BY 4.0). That license allows this personal, non-commercial project, including adapting the files, as long as Inside Airbnb is credited and changes are noted. The files here are filtered and given new columns; they are not a verbatim republication of the full dump. Source: [insideairbnb.com/get-the-data](https://insideairbnb.com/get-the-data/) and [insideairbnb.com/data-policies](https://insideairbnb.com/data-policies/). Inside Airbnb is not Airbnb and is not affiliated with this project.

We did not scrape Airbnb.com.

A few sources were considered and not used:

- **The calendar file.** Inside Airbnb says the calendar does not tell a booked night from a night the host blocked. Treating “unavailable” as “booked” would invent occupancy. We left the calendar alone.
- **Census tract income.** Listings in these files have no ZIP code, and a tract join needs a boundary file we would have to maintain. Neighbourhood and coordinates already carry the local price level. Adding Census would have been a second geography project.
- **OpenStreetMap extracts.** Distance to the airport uses a handful of published FAA airport reference points typed into `data/cities.py`, not a map download.
- **Redfin listing rows.** Redfin’s terms forbid automated collection, and the MLS owns the listing data. There is no downloader in this repo. The optional CSV upload in the local app is for a file you downloaded yourself. It is not part of the study.

### Why only three cities

Austin, San Diego, and Seattle are each one metro. Their listings sit in a tight patch of coordinates (latitude spread on the order of 0.05–0.07 degrees), the snapshots are from 2025, and bedrooms are almost filled in.

After the price window below, Austin keeps 10,296 listings and was missing a price on only 16 rows. San Diego keeps 11,590. Seattle keeps 5,992, and only 2 listings sat above the $2,000 cap, so a luxury tail does not run the score.

The other snapshots are still on disk and are not in these results.

- Hawaii is several islands in one file. A single “city” model would mix markets that are an ocean apart.
- Las Vegas, Fort Lauderdale, and Nashville are missing a nightly price on roughly 30% of rows. The priced subset may be a biased slice of hosts who bother to publish a rate.
- Asheville’s bedrooms column is essentially empty, so a size model has nothing to stand on.
- The remaining cities are smaller or have the same missing-price problem, and fitting fourteen markets would have made the evaluation thinner rather than better.

## Every cleaning step

Cleaning is in `data/ingest.py`. The rules are fixed dollar amounts and missingness checks. They are not percentiles of the whole file. A percentile computed before the split would let the test listings help decide who counts as an outlier.

Starting from the raw file, in order:

1. Keep every room type (entire home, private room, shared room, hotel room). Room type is a feature, and the baseline also gets room type, so the model is not praised for a split a median already makes.
2. Parse the price text to a number. Drop missing or non-positive prices.
3. Drop prices under $25. Those are placeholders and errors, not a real nightly rate.
4. Drop prices over $2,000. Above that, a few luxury or junk listings dominate squared error. This is a removal, not a cap: we do not quietly pull a $5,000 night down to $2,000 and then score the pulled value.
5. Drop rows with no latitude or longitude.
6. Drop rows with no bedroom count, or with a bedroom count outside 0–10. Zero is a studio and stays.
7. Drop rows with no room type. None of the three cities lost a row here.
8. If guest capacity (`accommodates`) is missing or zero, fill it with two guests per bedroom. That is a hole in the file, not a guess from other listings’ prices. Observed capacities are left alone.

Counts removed:

| Rule | Austin | San Diego | Seattle |
| --- | ---: | ---: | ---: |
| Rows in the file | 10,533 | 12,991 | 6,772 |
| Missing or non-positive price | 16 | 1,287 | 764 |
| Price under $25 | 72 | 3 | 2 |
| Price over $2,000 | 122 | 88 | 2 |
| Missing bedrooms | 13 | 19 | 10 |
| Bedrooms outside 0–10 | 14 | 4 | 2 |
| Rows used | 10,296 | 11,590 | 5,992 |

Room mix after cleaning: Austin 8,972 entire homes, 1,285 private rooms, 32 hotel rooms, 7 shared rooms. San Diego 9,898 / 1,505 / 147 / 40. Seattle 4,978 / 976 / 27 / 11.

Square footage is not on these files. An earlier assessor join was never saved with the snapshots, so this study does not invent it.

## The features, and why each one is there

The model that was locked in uses ten columns:

- **Bedrooms, bathrooms, accommodates.** Size is the obvious price driver. Capacity is not the same thing as bedrooms: a two-bedroom that sleeps eight is a different product.
- **Room-type flags** (`is_private_room`, `is_shared_room`, `is_hotel_room`). An entire home is the baseline, all three flags off. A private room should not be scored as if it were the whole house.
- **Latitude and longitude.** Price changes across a city in ways a single “distance to downtown” number cannot capture. In San Diego the coast is a price, and longitude shows up in the SHAP chart for that reason.
- **Distance to the city center, and distance to the nearest major airport.** Center distance is “how far from the tourist core.” Airport distance separates a neighborhood near the field from one that is not. These are computed from the listing’s coordinates and from fixed reference points. They do not use anyone’s price.

We also built, and then measured, features we did not keep. The rule for keeping a group is in the next section. The groups, in the only order we considered, were:

1. Size and room type, as above.
2. Plus location (coordinates and the two distances).
3. Plus property type: house, cabin, cottage, condo, townhouse, guest unit. A cabin and a condo are different buildings even when both are “entire place.”
4. Plus interactions:
   - accommodates × private room, and accommodates × shared room, because an extra guest does not change the price of a room the way it changes the price of a whole home;
   - distance-to-center × house, and distance-to-center × condo, so the drive from downtown can have a different slope by building type;
   - a smoothed mean log price of the neighbourhood, fit on the training rows only;
   - the same smoothing for neighbourhood × bedroom bin, because a one-bedroom and a four-bedroom in the same neighbourhood are different markets;
   - the median log price of the 40 nearest *other* training listings.
5. Plus listing quality: minimum nights (capped at 90 so a 365-night typo cannot dominate), amenity flags and a count, review score, log review count, reviews per month, superhost, log of the host’s listing count, instant book.

The neighbourhood and nearest-neighbour features use other listings’ prices. They are fit inside each training split and never see the row being scored, including when two listings share a coordinate. That is `features/fold.py`. Building them on the whole city and then splitting would leak the test price into the features.

Why listing quality is tempting and dangerous: reviews and amenities are real information about an ad that already exists. They are not known for a house that has never been listed. We measured them anyway, on the development splits, and the pre-set rule did not keep them. Details are under “What we tried that did not help.”

## How the data was split

This is the part that makes the numbers defensible.

**A cell, not a row.** Each listing is assigned to a grid cell of 0.03 degrees (about 3 km of latitude) from its coordinates alone. The cell id does not use the price.

**One locked test.** `GroupShuffleSplit` with seed 42 holds out 20% of the cells. That fraction is of cells, not of rows, so the row share is not exactly 20%: Austin 20.3% (2,087 listings), San Diego 27.1% (3,142), Seattle 16.9% (1,012). Larger cells make San Diego’s test heavier. Those listings were not used to pick features, tree depth, or XGBoost versus LightGBM. They were scored once, at the end.

**Same host cannot sit on both sides.** If a host has a listing in the test cells, their other listings are removed from training. A host’s pricing habits are not a free feature.

**Nothing within 1 km of a test listing stays in training.** A cell boundary is an arbitrary line. Without this buffer, the house across the street could still be in the training set. The buffer deletes training rows; it does not move them into the test.

What that costs, because it is a lot and should be visible:

| City | Training rows before the blocks | Removed for a shared host | Removed by the 1 km buffer | Training rows actually used |
| --- | ---: | ---: | ---: | ---: |
| Austin | 8,209 | 1,360 | 1,546 | 5,303 |
| San Diego | 8,448 | 2,421 | 2,193 | 3,834 |
| Seattle | 4,980 | 950 | 1,126 | 2,904 |

**Tuning never sees the test.** The development rows (the training side above) are split the same way five times, seeds 0 through 4: 20% of the remaining cells held out, then the same host purge and 1 km buffer. Mean MAE across those five seeds, averaged over the three cities, makes every modeling choice. We report that mean and its standard deviation. The standard deviation is the sample standard deviation across the five seeds, so it is a spread, not a confidence interval from a formula.

**Early stopping is inside the training side of whatever split is being fit.** A further 20% of those training cells pick the number of trees. That slice is not the validation fold and not the locked test. During the comparison, each candidate is scored from that early-stopped fit, so every candidate is treated the same. The single model that scores the locked test is then refit on all of the development rows, with the tree count early stopping chose. The saved file in `models/artifacts/` is that development refit. The test listings are not in it.

**The target is log(1 + price).** Nightly rates are skewed. Squared error on dollars spends the model on a few expensive listings. Squared error on the log treats a 20% miss on a $100 night like a 20% miss on a $400 night. Every reported R², MAE, and RMSE is computed after converting predictions back to dollars with `expm1`. We do not report a log-space R² and call it the result.

## The models we tried

All of these were fit on the development splits. None of them were chosen by looking at the locked test.

**Baseline.** Median training price for the same neighbourhood and room type. If that pair was not in the training cells, fall back to the neighbourhood median, then to the overall training median. A stricter baseline adds a bedroom bin (0 through 6) on top of neighbourhood and room type.

**Ridge regression** on the same columns as the selected tree model, including any fold-fit encodings when those were in the candidate set. Medians for missing values and the scaling are fit on the training side of that split only. The penalty is chosen by `RidgeCV` on that same training side, over a fixed grid of alphas. The target is the log price; the score is in dollars.

**XGBoost** with a histogram tree builder. Learning rate 0.05, `min_child_weight` 8, row and column subsampling 0.8, and L2 penalty 1 were fixed before looking at the other cities. The only depth values tried were 4, 6, and 8. Up to 400 trees, with early stopping after 30 rounds of no improvement.

**LightGBM** at the depth XGBoost had already settled on, same learning rate and similar leaf constraints. It replaces XGBoost only if its mean development MAE is under 98% of XGBoost’s. That bar was written down so a 1% wobble could not change the headline library.

### What was selected, and the rule

Feature groups are cumulative. We take the simplest prefix whose mean development MAE, averaged across cities and seeds, is within 1% of the best prefix. Shallower trees win the same way.

Development MAE, averaged across the three cities and five seeds:

| Feature prefix | Mean MAE |
| --- | ---: |
| Size and room type | $70.04 |
| Plus location | $68.02 |
| Plus property type | $67.78 |
| Plus interactions and spatial comps | $75.76 |
| Plus listing quality | $67.51 |

Listing quality is the best by a hair ($67.51 versus $68.02 for location). That is a 0.7% gap, inside the 1% band, so the simpler location model is the one we kept. Property type is in the same band and was not worth the extra columns either.

Tree depth, on that location model, same average:

| Depth | Mean MAE |
| --- | ---: |
| 4 | $68.13 |
| 6 | $68.02 |
| 8 | $68.16 |

Depth 6 is best. Depth 4 is within 1%, so the selected model is **XGBoost with depth 4**.

LightGBM at that depth had mean development MAE $67.49, against $68.13 for XGBoost. That is about a 0.9% improvement, short of the 2% bar. The headline model stays XGBoost.

Per city, on the five development seeds, for the model we actually kept (mean ± standard deviation):

| City | Model | R² | MAE | RMSE |
| --- | --- | ---: | ---: | ---: |
| Austin | Neighbourhood × room | -0.031 ± 0.034 | 103.00 ± 15.79 | 214.12 ± 30.17 |
| Austin | × bedroom bin | 0.029 ± 0.121 | 97.76 ± 17.14 | 207.79 ± 34.90 |
| Austin | Ridge | 0.118 ± 0.424 | 78.94 ± 4.86 | 190.73 ± 21.57 |
| Austin | XGBoost | 0.385 ± 0.119 | 71.90 ± 5.93 | 163.29 ± 10.98 |
| San Diego | Neighbourhood × room | -0.234 ± 0.310 | 144.23 ± 39.54 | 242.36 ± 57.42 |
| San Diego | × bedroom bin | -0.111 ± 0.324 | 134.60 ± 34.16 | 227.67 ± 47.06 |
| San Diego | Ridge | 0.531 ± 0.062 | 78.72 ± 15.96 | 149.72 ± 32.67 |
| San Diego | XGBoost | 0.480 ± 0.157 | 85.18 ± 23.42 | 157.70 ± 44.35 |
| Seattle | Neighbourhood × room | -0.083 ± 0.072 | 70.90 ± 13.57 | 125.04 ± 29.65 |
| Seattle | × bedroom bin | -0.079 ± 0.071 | 70.63 ± 13.50 | 124.79 ± 29.72 |
| Seattle | Ridge | 0.458 ± 0.146 | 49.15 ± 11.28 | 88.54 ± 27.03 |
| Seattle | XGBoost | 0.454 ± 0.155 | 47.31 ± 12.07 | 89.19 ± 29.02 |

Read the San Diego row carefully. On the development splits, ridge had a lower MAE than the tree model ($78.72 versus $85.18). We did not switch to ridge, because the protocol’s selected model is the tree, and ridge is the linear baseline. The locked test, below, went the other way. Both facts stay in the write-up.

The negative R² on the neighbourhood median is not a bug in the median. The test cells are different parts of the city. A neighbourhood the training cells never saw falls back to a citywide median, and a citywide median is a bad guess for a held-out beach or a held-out suburb. That is what a spatial split is supposed to reveal.

The chart is `reports/figures/model_comparison.png`.

### Learning curves

On development seed 0 only — still not the test — we refit the selected model on 15%, 30%, 50%, 75%, and 100% of the training cells and scored the same validation cells. The chart is `reports/figures/learning_curves.png`.

Validation MAE generally falls as more of the city is available, and the train MAE and validation MAE end up close once the full development training cells are used. San Diego overfits when only a small slice of cells is used (train MAE around $40 against validation MAE around $80) and the gap closes by the time most cells are in. The curves are jagged because a random subset of cells is not a smooth “more data” knob: one unlucky set of cells can be worse than a smaller set. That noise is why the curve is a diagnostic, not a tuning tool.

## The locked test

Scored once, after the choices above.

| City | Model | R² | MAE | RMSE |
| --- | --- | ---: | ---: | ---: |
| Austin | Neighbourhood × room | -0.688 | $152.46 | $282.59 |
| Austin | × bedroom bin | -0.395 | $138.47 | $256.92 |
| Austin | Ridge | 0.026 | $111.00 | $214.67 |
| Austin | XGBoost | 0.426 | $95.61 | $164.78 |
| San Diego | Neighbourhood × room | 0.033 | $135.52 | $251.82 |
| San Diego | × bedroom bin | 0.339 | $110.50 | $208.29 |
| San Diego | Ridge | 0.461 | $90.40 | $187.99 |
| San Diego | XGBoost | 0.555 | $85.02 | $170.79 |
| Seattle | Neighbourhood × room | -0.029 | $70.51 | $128.60 |
| Seattle | × bedroom bin | -0.031 | $70.36 | $128.70 |
| Seattle | Ridge | 0.347 | $51.27 | $102.47 |
| Seattle | XGBoost | 0.437 | $49.80 | $95.16 |

On this one split the tree model beats the median and beats ridge in every city. Austin’s ridge basically fails (R² 0.026) while the tree still reaches 0.426, which matches the SHAP picture: Austin’s price is a spatial surface, and a linear model on raw latitude and longitude is a poor surface.

Entire homes only, same predictions, same test cells (the old project’s population, reported as a slice rather than a different experiment):

| City | Entire-home test R² | MAE | RMSE | n |
| --- | ---: | ---: | ---: | ---: |
| Austin | 0.413 | $101.31 | $172.48 | 1,750 |
| San Diego | 0.583 | $84.06 | $159.28 | 2,689 |
| Seattle | 0.423 | $53.15 | $104.41 | 752 |

An earlier audit of the previous pipeline, on a random 20% of listings and on entire homes, put honest R² roughly in the 0.10–0.53 range. This test is a harder question (whole cells held out, hosts blocked, a 1 km gap), and entire-home R² lands at 0.413, 0.583, and 0.423. It is not the same split, so it is not a paired “before and after” on identical rows. It is the number you get when next-door listings are not allowed to help.

Predicted versus actual, and residual versus predicted, are `reports/figures/pred_vs_actual.png` and `reports/figures/residuals.png`.

### Where the errors are

Dollar error grows with the predicted price. That is what a log target does: it cares about relative error, so a miss on an expensive listing is a lot of dollars.

Austin, MAE by predicted-price quintile: about $50, $65, $82, $87, then $195 in the top bin. The top bin is predictions from about $283 to $1,106.

San Diego: about $26, $41, $58, $95, then $206. The mean residual in that top bin is +$84, which means the model is still low on the listings it already thinks are expensive. It shrinks toward the middle.

Seattle’s top bin (predictions up to about $569) has MAE about $102. The cheaper four bins sit between $32 and $46.

Room type on the same test:

- Entire homes carry most of the R² above.
- Private rooms are cheaper and the dollar MAE is smaller (Austin $60.64, San Diego $74.27, Seattle $40.03), but Seattle’s private-room R² is negative: the model is not beating the private-room mean there.
- Hotel rooms are rare (11 in Austin’s test, 39 in San Diego, none in Seattle’s test) and the model misses them badly. San Diego hotel-room MAE is $280.
- Shared rooms are a handful of listings. An R² on 2 or 6 or 12 rows is noise. Ignore it.

### What the model is using

SHAP values here are the average absolute contribution to the log-price prediction, on a sample of up to 600 locked-test rows. They explain the model we scored. They were not used to drop features after the fact. The chart is `reports/figures/shap.png`.

- **Austin:** distance to downtown (0.32), bedrooms (0.21), bathrooms (0.17), then longitude, capacity, and latitude. Private-room status is real but smaller (0.05). Airport distance is minor (0.02).
- **San Diego:** guest capacity (0.26) and longitude (0.14) lead, then bedrooms and bathrooms. Longitude is the coast. Downtown distance matters less than in Austin.
- **Seattle:** bedrooms (0.19), capacity (0.16), and private-room status (0.09), then distance to downtown.

## The old R² near 0.97

The previous training target was not the listing’s price. Listings were clustered geographically, grouped by cluster and bedroom count, and each row’s target was replaced by that group’s median price. The model was then scored against those medians. Medians are smooth. A model that has latitude and bedrooms can redraw a smooth median easily.

On Austin, as a demonstration and not as a result we are claiming (it uses the whole city, including what would have been a test):

- A model trained on the group median and scored against that median has cross-validated R² **0.973** and MAE about **$7**.
- The same predictions, scored against the listing’s own price, have R² **0.385** and MAE **$84.40**.

That is the inflation. The second number is in the same ballpark as a real model. The first number is the target being easy.

A separate, smaller issue is that a median computed before the split includes the listing’s own price. On Austin the correlation between price and the median that includes the listing is 0.648. The correlation with a leave-one-out median is 0.635. So “the row helped write its own target” is real but modest. Scoring the smoothed target as if it were the listing price is the thing that manufactures a 0.97.

This study does not train on a median. The baselines use medians as predictions. The model’s target is the listing.

## What we tried that did not help

The ablation chart is `reports/figures/ablation_mae.png`. Development MAE by city (mean ± sd across five seeds), XGBoost depth 6, which is the depth the ablation used before the depth search:

| Prefix | Austin | San Diego | Seattle |
| --- | ---: | ---: | ---: |
| Size and room | 76.57 ± 4.43 | 85.11 ± 14.13 | 48.44 ± 11.99 |
| Plus location | 72.38 ± 6.21 | 83.74 ± 21.85 | 47.94 ± 12.19 |
| Plus property type | 72.03 ± 5.49 | 82.93 ± 20.06 | 48.39 ± 12.72 |
| Plus interactions | 76.10 ± 6.37 | 94.20 ± 19.79 | 56.99 ± 14.45 |
| Plus listing quality | 67.52 ± 6.03 | 83.08 ± 15.47 | 51.93 ± 13.07 |

Location helps in all three cities, a few dollars of MAE. Property type does not move the average enough to keep.

**Interactions and nearest-neighbour prices made things worse** under this split, by about $4 in Austin, $10 in San Diego, and $9 in Seattle versus location alone. The reason is the split itself. The 40 nearest training listings are at least 1 km away, often in a different cell, so “comps” are not comps. A neighbourhood encoding fit only on other cells does not know the held-out neighbourhood and falls back to a broad mean. Fed to a tree, that noise hurts. On a random split that lets the house next door into training, those features would look strong. That strength would be the leakage this design was built to refuse.

**Listing quality helps Austin and hurts Seattle.** Austin’s development MAE falls from $72.38 to $67.52 when reviews, amenities, and host fields are added. Seattle’s rises from $47.94 to $51.93. Averaged across cities the gain is under 1%, so the fields stay out. That is also the honest product statement: once you cannot see the listing next door, the ad’s own reviews are not a reliable extra, and they are unavailable for a house that has no ad yet.

**LightGBM** was a little better on the development average and not better enough to switch. **Deeper trees** were the same story.

**A pooled model across cities** was not fit. Three cities with a pre-committed per-city protocol was the study. Pooling would have been a second project and a way to shop for a better number.

## What the local app is doing

The Flask app and the React screen are a way to look at the three cities and at a single scored property. They are not a second evaluation.

- The number the model owns is one nightly rate.
- The 12-month curve multiplies that rate by hand-set seasonal weights in `data/cities.py`. The weights average to 1, so they do not change the annual average. They are a sketch of the travel calendar (SXSW, summer, and so on), not a fit to bookings.
- Occupancy on the screen is the median, in that neighbourhood, of Inside Airbnb’s San Francisco review proxy, computed on the development listings only. The proxy assumes half of stays leave a review, uses the city’s average stay or the listing’s minimum nights when that minimum is longer, divides by 30, and caps the result at 70%. Listings with no reviews are left missing rather than called 0% occupied. We did not train a model to predict this proxy. The inputs that define it are the review counts, and using those inputs would be circular. The calendar cannot supply a cleaner label.
- The 0–100 score is a mortgage worksheet on top of that revenue. It is not a measure of predictive accuracy.

For a home that is not an existing listing, reviews and amenities are left missing. They are not filled with zeros. Zeros would mean “no kitchen” and “no reviews.”

## Limitations

- The label is the advertised rate, not what a guest paid and not what a host cleared after fees.
- Airbnb jitters coordinates by up to about 150 meters. Distances and cells are fuzzy at that scale.
- The 1 km buffer and the host purge throw away a large share of training rows. The model is answering “a part of the city I did not train on,” which is harder than “the house next door,” and it is also a smaller training set.
- Test size differs by city because cells are not equal. San Diego’s test is 27% of rows.
- Hotel rooms and shared rooms are too rare to judge.
- Errors of $50 to $200 a night are large relative to the price. The model is a ranking and a rough level, not a quote.
- Expensive listings are systematically a bit under-predicted.
- Development seed-to-seed MAE moves by $6 in Austin and by $20 in San Diego. One split’s MAE is not a stable law.
- Austin’s development R² for ridge has a standard deviation of 0.42. One bad spatial fold swings a linear model hard. Do not quote a development R² without the spread.
- The saved model has not seen the test listings. That is intentional. It is slightly weaker than a model refit on every row would be, and the test number matches the model that was saved.
- Three cities, one protocol. Nothing here is a national Airbnb model.
- Histogram trees were fit with every CPU core. A repeat run can move the third decimal of a metric. The tables match `reports/locked_test_predictions.csv` and `reports/metrics.json` from this run; those files are the record.

## How to reproduce

From the repository root, with the pinned `requirements.txt`:

```bash
pip install -r requirements.txt
python -m evaluation.run_eval
pytest
```

The evaluation refits every development split. It takes a while. It writes `reports/metrics.json`, `reports/locked_test_predictions.csv`, the figures, and `models/artifacts/<city>/model.joblib`.

The local app, after that:

```bash
cd backend && python app.py
```

and, in another terminal, `npm install && npm run dev`. The app only offers Austin, San Diego, and Seattle.
