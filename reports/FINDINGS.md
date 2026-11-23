# What CapScout found

CapScout asks two questions about a home in Austin, San Diego, or Seattle. What nightly rate could it advertise on Airbnb? About how many nights a year would it book? The app multiplies those two numbers and gives the result a letter grade inside the city.

This is a personal project. The rate is the advertised rate, not the price a guest paid. The nights number is built from reviews, not from a calendar of bookings. A separate effort is collecting daily calendars for a real occupancy label. This study does not replace that work, and it does not pretend the review proxy is a booking count.

## The data

The rows are Inside Airbnb listing snapshots, CC BY 4.0, credit Inside Airbnb. Austin is the 2025-09-16 snapshot. San Diego and Seattle are the 2025-03-16 snapshots. The files are downloaded with `python -m data.download austin san-diego seattle` and are not committed. Prices that look like `$1,234.00` are parsed by removing the dollar sign and the comma. A bathroom written as "Half-bath" with no digit is 0.5. On these three files, after the price filters, the numeric bathroom column was already filled, so that rule changed zero kept rows. It is still the parser.

Every cut below is a fixed rule, not a percentile of the file. Counts are sequential.

| Filter | Austin, TX removed | Austin, TX left | San Diego, CA removed | San Diego, CA left | Seattle, WA removed | Seattle, WA left |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rows in the Inside Airbnb file | 0 | 10,533 | 0 | 12,991 | 0 | 6,772 |
| Duplicate listing id (kept the copy with more reviews in the last year) | 0 | 10,533 | 0 | 12,991 | 0 | 6,772 |
| Price missing, not a number, or not positive | 16 | 10,517 | 1,287 | 11,704 | 764 | 6,008 |
| Price under $25 | 72 | 10,445 | 3 | 11,701 | 2 | 6,006 |
| Price over $2,000 | 122 | 10,323 | 88 | 11,613 | 2 | 6,004 |
| Missing latitude or longitude | 0 | 10,323 | 0 | 11,613 | 0 | 6,004 |
| More than 60 km from downtown | 0 | 10,323 | 0 | 11,613 | 0 | 6,004 |
| Missing room type | 0 | 10,323 | 0 | 11,613 | 0 | 6,004 |
| Missing bedrooms | 13 | 10,310 | 19 | 11,594 | 10 | 5,994 |
| Bedrooms outside 0 to 10 | 14 | 10,296 | 4 | 11,590 | 2 | 5,992 |
| Bathrooms outside 0 to 12 | 0 | 10,296 | 0 | 11,590 | 4 | 5,988 |
| Guest capacity outside 1 to 16 | 0 | 10,296 | 0 | 11,590 | 0 | 5,988 |
| Minimum stay over 90 nights | 55 | 10,241 | 31 | 11,559 | 32 | 5,956 |
| No reviews in the last 12 months | 2,422 | 7,819 | 2,995 | 8,564 | 1,272 | 4,684 |
| Unavailable every night of the year | 41 | 7,778 | 104 | 8,460 | 61 | 4,623 |
| Same host, spot, room type, bedrooms, and price (kept the reviewed copy) | 58 | 7,720 | 73 | 8,387 | 13 | 4,610 |
| Price or review-based nights could not be computed | 0 | 7,720 | 0 | 8,387 | 0 | 4,610 |

Austin goes from 10,533 rows to 7,720. San Diego goes from 12,991 to 8,387. Seattle goes from 6,772 to 4,610. The farthest kept home is 30.0 km from downtown Austin, 42.9 km from downtown San Diego, and 14.5 km from downtown Seattle. The 60 km rule dropped nobody. It is there so a swapped coordinate or another city cannot sneak in. San Diego's north-county listings stay.

Listings with no review in the last 12 months are removed, and so are listings marked unavailable every night of the next year. A missing availability value is kept. Near-duplicate ads from the same host, at the same rounded coordinate, with the same room, bedrooms, and price, keep the copy with more recent reviews.

Census ACS 2024 5-year income, rent, home value, and population density were joined on tract polygons (public domain, TIGER/Line 2024). 99.8% of Austin rows, 100.0% of San Diego rows, and 99.9% of Seattle rows fell inside a tract. A missing tract statistic is the city median of the tracts, not a price. OpenStreetMap was not added. An earlier round measured it and it did not help. This round did not rerun that test.

An older pipeline trained on area-median prices. As a demonstration, that fit scored about 0.973 against the smoothed median and about 0.385 against the listing's own price. This study trains on the listing's own price.

## How the test was locked

Before any model was fit, 20% of the 3 km grid cells were held out with seed 17. The cell list is sorted before the draw, so row order cannot change who is held out. Every listing in a held-out cell is held out. Training then drops any host who also has a held-out listing, and any training listing within 1 km of a held-out listing. The test side is not trimmed. The held-out ids are in `evaluation/study_ids.json`. That file has ids only. A later run that does not match it stops.

Checked on the primary split: no shared cell, no shared host, and the closest training and test homes are at least 1 km apart. Austin's test is 2,101 listings in 31 cells, with 3,281 left to train. San Diego's test is 817 listings in 23 cells, with 5,571 left to train. Seattle's test is 730 listings in 8 cells, with 2,585 left to train.

Features and the small hyperparameter grids were chosen with 5-fold grouped cross-validation on the training cells only. The primary holdout was scored once. The simplest eligible model within 2% of the best holdout error is the one that ships. The eligible models are the local median rate, linear regression, ridge regression, a random forest, and XGBoost.

Seeds 19, 29, and 37 are a second check. Same features, same settings, new cells. They were not used to change the choice.

## What went into the model

The nightly rate started from a short list: bedrooms, bathrooms, beds, guests, whether the bath is shared, room type, house versus condo, latitude, longitude, distance to downtown, distance to the airport, a handful of amenities (count, air conditioning, parking, washer, kitchen, pool, hot tub, waterfront), review score, log of the review count, and superhost. Unknown superhost stays missing. It is not coded as "no." After ridge was chosen, three more columns were added under the rule in the feature section below: a public-place index, distance to the coast in San Diego and Seattle, and bedrooms times guests.

A probe XGBoost on the training folds had average error of about $60.03 with the home only, $58.24 after amenities, $55.43 after reviews, and $54.89 after census. Census was the best. The 2% band around it still included the reviews list, which is shorter, so census stayed out.

The nights model uses the home and the amenities, and not the reviews. Reviews, minimum stay, price, and availability build the label or are unknown for a house that has never been listed. Census again helped by less than 2% on the training folds (about 71.57 nights versus 71.84), so it stayed out.

The nights target stays in nights, not log nights. A log target was worse on the training folds (73.76 versus 71.84).

The nights models predict nights directly. On the training folds a log target averaged 73.76 nights of error against 71.84 for the direct target, which is not a 2% improvement, so the log target was not used.

## The nightly rate

Every model below was scored once on the primary holdout, on the columns before the added features. The average gives each city equal weight. The local median rate is the training median for that neighbourhood and room type, with a fallback when fewer than 8 training listings support the group.

| Model | Austin, TX | San Diego, CA | Seattle, WA | Average |
| --- | ---: | ---: | ---: | ---: |
| Local median rate | $112.75 | $91.62 | $49.40 | $84.59 |
| Linear regression | $73.26 | $56.89 | $36.26 | $55.47 |
| Ridge regression | $72.13 | $57.27 | $36.13 | $55.18 |
| Random forest | $72.09 | $55.98 | $35.92 | $54.67 |
| XGBoost | $71.81 | $53.88 | $36.89 | $54.19 |

The lowest average error was $54.19 a night, from XGBoost. Ridge regression was $55.18, inside the 2% band (up to $55.28), and it is the simplest model in that band.

Ridge is linear regression with a penalty that shrinks the weights. The penalty was chosen on the training folds, preferring the stronger penalty when it was within 1% of the best. Austin and San Diego use 100. Seattle uses 10. The model predicts the log of the rate and is converted back to dollars. Predictions are floored at zero and are not capped, so an expensive guess stays visible.

XGBoost is a little better in San Diego ($53.88 versus $57.27) and essentially tied in Austin and Seattle. Across the three cities the gap is about one dollar. That is inside the 2% rule, so the simpler model ships.

Primary holdout on those columns, with a 95% interval from resampling the held-out cells 1,000 times.

| City | Listings | Cells | R² | R² interval | MAE | MAE interval | Median % error | Median % interval | Local median MAE | Error removed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin, TX | 2,101 | 31 | 0.511 | 0.449 to 0.665 | $72.13 | $56.38 to $85.13 | 25.9% | 22.7% to 28.1% | $112.75 | 36.0% |
| San Diego, CA | 817 | 23 | 0.601 | 0.517 to 0.673 | $57.27 | $45.40 to $75.88 | 21.5% | 19.2% to 24.5% | $91.62 | 37.5% |
| Seattle, WA | 730 | 8 | 0.522 | 0.489 to 0.592 | $36.13 | $31.43 to $40.24 | 19.9% | 17.3% to 22.4% | $49.40 | 26.9% |

## The shipped holdout

The headline is linear regression on the adopted columns, scored once on the same frozen holdout. Ridge was the fit right after the feature bundle. A later check, with the hyperparameters chosen on the training cells, scored the eligible models once. Equal-city average miss: linear regression $53.63, XGBoost $53.67, ridge $53.95, random forest $54.37, local median rate $84.59. The 2% band goes up to $54.70, so linear regression ships. Three further columns, the log of distance to downtown, guests in a private room, and bathrooms per guest, moved the training-cell error from $56.01 to $55.58 and were not added.

Counting every home once, R² is 0.571 (0.518 to 0.665), the median error is 23.0% (20.5% to 25.4%), and the average miss is $59.80 ($49.01 to $69.20) against $95.34 for the local median rate. That is 37.3% less error (34.2% to 42.2%). The holdout is 3,648 homes. Seattle's part of the interval resamples only 8 held-out cells.

| City | R² | Average miss | Median error | Less error than the local median rate |
| --- | ---: | ---: | ---: | ---: |
| Every home once | 0.571 | $59.80 | 23.0% | 37.3% |
| Austin, TX | 0.540 | $70.00 | 25.7% | 37.9% |
| San Diego, CA | 0.641 | $54.48 | 19.9% | 40.5% |
| Seattle, WA | 0.536 | $36.41 | 19.8% | 26.3% |
| Each city once | 0.572 | $53.63 | 21.8% | 36.6% |

Austin's interval is R² 0.484 to 0.671 and a miss of $54.47 to $83.96, on 2,101 homes in 31 cells. San Diego is R² 0.565 to 0.698 and $44.09 to $71.66, on 817 homes in 23 cells. Seattle is R² 0.494 to 0.600 and $31.01 to $42.32, on 730 homes in 8 cells. The measured coverage of the range is 78.4% in Austin, 81.0% in San Diego, and 79.7% in Seattle. Median width is about $147, $107, and $87. The band was not widened after that look.

Counting each city once, the shipped miss is $53.63 against $84.59, which is 36.6% less error (33.7% to 41.7%).

The same linear specification on seeds 19, 29, and 37. These seeds were not used to pick the model. San Diego's R² on one of them is −0.184. The dollar miss on San Diego's three seeds stays between $63.62 and $71.22.

| City | Primary R² | Other seeds R² | Primary MAE | Other seeds MAE |
| --- | ---: | ---: | ---: | ---: |
| Austin, TX | 0.540 | 0.502, 0.452, 0.586 (spread 0.134) | $70.00 | $51.91, $60.83, $47.40 (spread $13.43) |
| San Diego, CA | 0.641 | 0.683, 0.652, -0.184 (spread 0.867) | $54.48 | $66.38, $71.22, $63.62 (spread $7.60) |
| Seattle, WA | 0.536 | 0.486, 0.569, 0.497 (spread 0.083) | $36.41 | $39.12, $34.94, $32.01 (spread $7.11) |

## The columns the model was chosen on

These paragraphs are the score before the added features. They are why ridge was chosen. They are not the shipped score above.

Counting every holdout home once is the pooled result on these columns. The miss is $61.60 a night, against $95.34 for the local median rate, which is 35.4% less error. R² is 0.542, and the median error is 23.6%. The holdout is 3,648 homes. Resampling cells inside each city 1,000 times, then stacking the draws, gives a 95% interval of R² 0.480 to 0.659, a miss of $50.26 to $71.34, a local-median miss of $79.54 to $109.27, a median error of 21.2% to 25.6%, and 32.1% to 41.0% less error. Seattle's part of that interval resamples only 8 held-out cells.

The city rows stay beside that pool. Austin, San Diego, and Seattle are different markets: the median advertised rate on this holdout is $146 in Austin, $153 in San Diego, and $111 in Seattle, and the holdouts are 2,101, 817, and 730 homes. One score for all three would let Austin's prices, and its larger sample, speak for the other two cities.

Counting each city once is the overall the model was chosen on. Ridge's average miss is $55.18 a night, against $84.59 for the local median rate, which is 34.8% less error. The average of the three R² values is 0.545, and the average of the three median percent errors is 22.5%. Resampling each city's holdout cells 1,000 times, then averaging the three cities, gives a 95% interval of R² 0.506 to 0.611, a miss of $48.86 to $63.45, a local-median miss of $77.07 to $100.80, a median percent error of 21.0% to 24.1%, and 31.6% to 40.4% less error.

| Summary | R² | Model miss | Local median rate | Median % error | Less error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Every home once | 0.542 (0.480 to 0.659) | $61.60 ($50.26 to $71.34) | $95.34 ($79.54 to $109.27) | 23.6% (21.2% to 25.6%) | 35.4% (32.1% to 41.0%) |
| Each city once | 0.545 (0.506 to 0.611) | $55.18 ($48.86 to $63.45) | $84.59 ($77.07 to $100.80) | 22.5% (21.0% to 24.1%) | 34.8% (31.6% to 40.4%) |

The same specification on seeds 19, 29, and 37:

| City | Primary R² | Other seeds R² | Primary MAE | Other seeds MAE |
| --- | ---: | ---: | ---: | ---: |
| Austin, TX | 0.511 | 0.455, 0.396, 0.574 (spread 0.178) | $72.13 | $54.41, $64.01, $49.60 (spread $14.41) |
| San Diego, CA | 0.601 | 0.650, 0.650, 0.182 (spread 0.468) | $57.27 | $69.37, $74.25, $62.31 (spread $11.94) |
| Seattle, WA | 0.522 | 0.465, 0.544, 0.492 (spread 0.079) | $36.13 | $39.79, $36.18, $32.46 (spread $7.33) |

Seattle is stable on those columns. Austin's other seeds are a bit easier in dollars than the primary, so quoting $72.13 is not flattering. San Diego's R² falls to 0.182 on one of the three seeds. Those seeds were not re-run after the features below were added. Do not quote San Diego's later 0.639 as if the other seeds had agreed.

The range on those earlier columns covered 79.0% of Austin's holdout, 82.0% of San Diego's, and 81.8% of Seattle's, with median widths about $155, $115, and $83. The shipped range, refit on the training folds of the adopted model, is the one in the section above.

## The pre-registered feature try

The rule, written into the scoring script before it printed a result: adopt the bundle only if grouped spatial cross-validation on the training cells improves the equal-city average error by at least 2%. The bundle is an inverse-distance attraction index from a fixed public list (six landmarks per city, distance floored at 0.25 km), distance to the nearest point on a fixed shoreline list in San Diego and Seattle, and bedrooms times guests. Austin has no coast, so that column is omitted there. Each piece is reported on its own and is not adopted by itself. The holdout is not used to choose.

The ridge penalties stay at the values already chosen: 100 in Austin, 100 in San Diego, and 10 in Seattle. Equal-city mean absolute error on the training folds:

| Columns | CV error | Change |
| --- | ---: | ---: |
| Shipped columns, before the bundle | $57.27 | |
| Attraction index alone | $56.54 | 1.3% lower |
| Coast distance alone | $57.33 | 0.1% higher |
| Bedrooms × guests alone | $56.73 | 0.9% lower |
| All three together | $56.01 | 2.2% lower |

The bundle clears 2%, so it is in the shipped model. By city, the bundle's training-fold error is $53.06 in Austin (from $55.14), $74.16 in San Diego (from $76.33), and $40.83 in Seattle (from $40.34). Seattle alone is a little worse. The rule uses the equal-city mean, and that mean cleared the bar. The holdout was then scored once for ridge. The later check in "The shipped holdout" replaced that fit with linear regression.

## Where the error sits

On the shipped holdout, the share of homes within 20% of the advertised rate is 44.4% for the linear model and 29.6% for the local median rate. Within 30% the shares are 62.6% and 42.3%. By city, within 20%: Austin 39.9% versus 27.5%, San Diego 50.1% versus 29.9%, Seattle 50.8% versus 35.3%. Within 30%: Austin 58.5% versus 38.3%, San Diego 67.9% versus 43.1%, Seattle 68.5% versus 52.9%.

Error by advertised-price tier, all 3,648 homes. Bias is prediction minus actual, so a negative number is an underprediction.

| Tier | Homes | Advertised rate | Average absolute error | Bias |
| --- | ---: | --- | ---: | ---: |
| Lowest quartile | 925 | $27 to $96 | $25 | +$21 |
| Second quartile | 912 | $97 to $138 | $30 | +$14 |
| Third quartile | 903 | $139 to $220 | $47 | +$3 |
| Highest quartile | 908 | $221 to $1,833 | $139 | −$81 |
| Top 10% | 365 | $368 to $1,833 | $229 | −$163 |

The model overpredicts the cheaper homes and underpredicts the expensive ones. The top 10% of homes account for 38.3% of the absolute error. The same pattern shows up in each city: the highest quartile's bias is about −$100 in Austin, −$67 in San Diego, and −$37 in Seattle.

Blanking review score, review count, and superhost at test time was measured on the ridge fit, before linear regression replaced it. Letting the imputer fill the training median changed that equal-city average error from $53.95 to $53.83. Austin moved by +$0.28, San Diego by −$0.66, and Seattle by +$0.03. Dropping those three columns and refitting ridge on the training cells only, with the adopted columns otherwise kept, raises the equal-city training-fold error from $56.01 to $56.95, about 1.7% worse. That is not a 2% improvement, so the columns stay. The holdout was not used to make that call, and the check was not repeated for the linear model.

## How the study evolved

Earlier pipelines used other seeds, including 0 through 4, 7, 11, 23, 31, and 42. Those seeds are now forbidden so a later run cannot quietly reuse a split that was already looked at. The code and reports from those pipelines are on the git tag `archive-pre-v7`, which points at the main commit before this cleanup. They are not in this tree.

Seed 17, and the held-out cell ids in `evaluation/study_ids.json`, were fixed before the choice among the local median rate, linear regression, ridge, random forest, and XGBoost. The confirmation seeds were not part of that choice. The feature bundle above was the one later change, and it was allowed only by the 2% training-cell rule.

## A later snapshot

Inside Airbnb has a newer listing file for each city: Austin 2026-09-17, San Diego 2026-09-20, and Seattle 2026-09-20. The shipped linear model was scored on rows that pass the same cleaning and whose listing id and host id are both absent from the training rows. The local median rate is the one fit on the training rows, not a median from the new file. This is a check, not a new model choice. The home counts and the local-median misses match the earlier ridge check on this same set of homes.

| City | Snapshot | Homes | R² | Average miss | Median error | Local median miss | Less error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin | 2026-09-17 | 4,794 | 0.290 | $145.10 | 40.4% | $198.70 | 27.0% |
| San Diego | 2026-09-20 | 3,855 | 0.453 | $133.81 | 32.2% | $196.09 | 31.8% |
| Seattle | 2026-09-20 | 3,093 | -0.013 | $124.59 | 43.2% | $155.47 | 19.9% |
| Every home once | | 11,742 | 0.336 | $135.99 | 38.8% | $186.46 | 27.1% |

A year later, on homes and hosts the model did not train on, the error is much larger and Seattle's R² is slightly negative. The model still has less average error than the training-set local median rate. Prices moved, and a stale neighbourhood median is a weak baseline. This is not the 0.571 from the frozen holdout.

## Booked nights

Same holdout, same 2% rule, simpler models only. The comparison is the average miss in nights.

| Model | Austin, TX | San Diego, CA | Seattle, WA | Average |
| --- | ---: | ---: | ---: | ---: |
| Local median nights | 72.0 | 76.5 | 72.8 | 73.8 |
| Linear regression | 68.9 | 70.2 | 68.7 | 69.3 |
| Ridge regression | 68.6 | 70.3 | 68.8 | 69.2 |
| Random forest | 67.9 | 70.5 | 68.7 | 69.0 |
| XGBoost | 69.0 | 69.9 | 68.7 | 69.2 |

The lowest average error was 69.0 nights, from Random forest. Linear regression was 69.3 nights, inside the 2% band (up to 70.4 nights), and it is the simplest model in that band.

The bar, written down before this run and unchanged from the previous nights study, is an R² of at least 0.10 in every city and an average miss at least 10% lower than the local median nights. Both have to hold.

| City | R² | R² interval | MAE (nights) | MAE interval | Median % error | Local median MAE | Error removed | Cleared |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin, TX | 0.014 | -0.013 to 0.055 | 68.9 | 64.8 to 70.8 | 54.2% | 72.0 | 4.4% | No |
| San Diego, CA | 0.118 | 0.042 to 0.144 | 70.2 | 66.6 to 71.4 | 39.9% | 76.5 | 8.1% | No |
| Seattle, WA | 0.062 | -0.006 to 0.120 | 68.7 | 64.8 to 70.8 | 40.0% | 72.8 | 5.7% | No |

The nights model did not clear the pre-set bar in every city. The bar was an average miss at least 10% lower than the local median nights, and an R² of at least 0.10. The app labels nights as a rough estimate.

San Diego clears the R² half (0.118, and the interval stays above zero) and misses the 10% error cut (8.1%). Austin and Seattle miss both. The other seeds do not rescue it. Their R² values stay near the primary ones, and the dollar-style spread in nights is only about 1 to 3 nights.

| City | Primary R² | Other seeds R² | Primary MAE | Other seeds MAE |
| --- | ---: | ---: | ---: | ---: |
| Austin, TX | 0.014 | 0.032, 0.033, 0.057 (spread 0.025) | 68.9 | 68.5, 65.8, 66.6 (spread 2.8) |
| San Diego, CA | 0.118 | 0.100, 0.095, 0.060 (spread 0.039) | 70.2 | 71.2, 70.8, 73.1 (spread 2.3) |
| Seattle, WA | 0.062 | 0.073, 0.063, 0.056 (spread 0.016) | 68.7 | 70.5, 71.7, 71.9 (spread 1.4) |

That is an honest failure of the label more than of the model. A linear model, a forest, and XGBoost all sit within about a night of each other, and all of them sit close to the local median nights. The home's size and amenities do not tell you the review-based nights. The app still shows the number, labeled as a rough estimate, because hiding it would pretend the question was not asked.

Yearly revenue, predicted rate times predicted nights, has holdout R² 0.287, 0.273, and 0.151 on the shipped model, and average misses of about $13,262, $13,984, and $9,170. The grade cuts are the 20th, 40th, 60th, and 80th percentiles of training revenue from that model. On the holdout the letter matched the review-based revenue's letter 33.9% of the time in Austin, 34.5% in San Diego, and 34.7% in Seattle. The grade is a rank of the rough revenue estimate. It is not a profit grade.

## What to say, and what not to say

Say that ridge was inside 2% of XGBoost on the columns used for the first choice, that three pre-registered features then cleared a 2% training-cell bar, and that a later check on those columns scored the eligible models once. Linear regression had the lowest equal-city error and is the simplest model inside 2% of it, so that is what ships. On that score, counting every home once, the average error is 37.3% lower than the local median rate and R² is 0.571. Say the test was geographic and frozen first. Say the nights figure is rough and missed a bar that was not moved. Say expensive homes are underpredicted. Say a newer snapshot of new homes and new hosts, scored with this linear model, has pooled R² 0.336 and a median error of 38.8%, and Seattle's R² there is −0.013.

Do not say the model knows the booked nights. Do not say the range is an 80% promise. Austin's measured coverage on the shipped model is 78.4%. Do not quote San Diego's 0.641 as if the other seeds agreed. On the shipped linear model those seeds run from −0.184 to 0.683. The earlier column list, where San Diego's R² ran from 0.182 to 0.650, is a different score. Do not quote XGBoost's pooled R². Its equal-city error did not beat linear by enough to leave the 2% band, and it was worse in Seattle.

## A calendar panel for a later label

The nights holdout above is a model of the review proxy. It is not evidence about real bookings, because the public calendar on one day does not say which unavailable nights were sold. The way to get a real label is to watch the same calendars until nights flip.

The collector takes a fixed panel from the newest Inside Airbnb listings for Austin (2026-09-17), San Diego (2026-09-20), and Seattle (2026-09-20). A listing is eligible when it has a review in the twelve months before that file and at least one open night. The panel draw uses seed 17, the same number as the rate study's primary holdout, and it is a sample of listings, not that holdout. It is stratified by neighbourhood and room type. The ids are `data/calendar_sample.json`. The eligible rows are `data/calendar_pool/`. Each day the job stores `listing_id`, `date`, `available`, `min_nights`, `price` when present, and `fetched_at`, gzipped, on the `calendar-data` branch. The client waits about 2 seconds between listings, backs off on ordinary server errors, and stops when it is blocked or challenged. It does not log in, solve a captcha, or use a proxy. If the public calendar cannot be read from the runner, the same panel is cut out of Inside Airbnb's published calendar file and the run report says that. One published file has no flip to measure, so that path does not produce a booked-night rate.

On 2026-11-19 a probe asked 36 eligible ids per city, with the ids read as text. GitHub Actions has not executed the workflow. The cron starts from the default branch after merge, so whether a hosted runner can fetch is still open. The public endpoint answered on the machine that prepared the change: Austin 35 of 36, San Diego 35 of 36, Seattle 36 of 36. Failed was 0, and the probe was not challenged. The daily ask is 309, 309, and 300 so the expected number of calendars is about 300. That pass is under an hour at the probe's pace, inside the 180-minute job.

The same day, a first panel of 300 ids from the 2025 study files returned 80, 93, and 92 calendars. Those files are the 2026-11-19 commit on `calendar-data`. They are retired. About 90 calendars a city is too thin for the 26-week label, which is why the panel was redrawn. The label command keeps the current sample. Listing ids stay text: rounding an 18-digit id through a float asks for a different home, and that calendar comes back empty.

The inference, in `data/calendar_labels.py`, follows the daily-scrape rule that a booking is an available night that later becomes unavailable, plus the lead-time and length cuts that separate a short guest stay from a host block:

- Booked: a contiguous run of 1 to 13 nights that was still open on a snapshot inside 45 days of the check-in, then closed, and at least as long as the posted minimum stay.
- Owner block: 14 or more nights already closed when the check-in was 90 days out. Ninety days is the shortest availability window hosts can set. The same label is used when half or more of the nights that were open close between two snapshots at most 3 days apart.
- Closed window: 21 or more unavailable nights running through the last date the snapshot publishes, never seen open.
- Left out of the booked count: nights that were already unavailable on the first snapshot, nights that open again before the date, nights with no snapshot within 2 days, and runs whose lead time or length falls between the two cuts.

The rate is booked nights divided by booked nights plus nights still open at the date. Blocks and censored nights are counted beside it. A one-night owner hold can be called a booking when the minimum stay is 1. A fortnight stay, and a stay booked more than 45 days ahead, stay uncertain. Price is stored and unused, because the public calendar often leaves it blank on open nights and closed nights alike.

The first retrain waits 26 weeks. Nights already shut on the first day are censored. A night first seen beyond the 90-day line is a resolved outcome only once that date arrives, which is about 13 weeks after collection starts. A listing-level rate near 30% needs about 80 offered nights before its standard error is near 5 points (0.3 times 0.7, divided by 0.05 squared, is 84). That is about 12 further weeks. 13 + 12 is 25, and 26 weeks pulls the window off a single peak. The target at that point is the observed rate over the tracked window, not an annualized copy of it. An annual count, comparable to the review proxy, waits for 52 weeks. Occupancy accuracy will be re-tested on a fresh spatial holdout once the 26 weeks exist: a new seed, not seed 17 and not seed 31, cells frozen before any fit, scored once. The R² figures in the nights table above are about the review proxy. They are not a forecast of that test.
