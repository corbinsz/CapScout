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

Features and the small hyperparameter grids were chosen with 5-fold grouped cross-validation on the training cells only. The primary holdout was scored once. The simplest eligible model within 2% of the best holdout error is the one that ships. A neural net is a comparison and cannot ship. CatBoost, and a 50/50 blend, ship only if they beat the simple model by 5% on the average of the three city errors.

Seeds 19, 29, and 37 are a second check. Same features, same settings, new cells. They were not used to change the choice.

## What went into the model

The nightly rate uses a short list: bedrooms, bathrooms, beds, guests, whether the bath is shared, room type, house versus condo, latitude, longitude, distance to downtown, distance to the airport, a handful of amenities (count, air conditioning, parking, washer, kitchen, pool, hot tub, waterfront), review score, log of the review count, and superhost. Unknown superhost stays missing. It is not coded as "no."

A probe XGBoost on the training folds had average error of about $60.03 with the home only, $58.24 after amenities, $55.43 after reviews, and $54.89 after census. Census was the best. The 2% band around it still included the reviews list, which is shorter, so census stayed out.

The nights model uses the home and the amenities, and not the reviews. Reviews, minimum stay, price, and availability build the label or are unknown for a house that has never been listed. Census again helped by less than 2% on the training folds (about 71.57 nights versus 71.84), so it stayed out.

The nights target stays in nights, not log nights. A log target was worse on the training folds (73.76 versus 71.84).

The nights models predict nights directly. On the training folds a log target averaged 73.76 nights of error against 71.84 for the direct target, which is not a 2% improvement, so the log target was not used.

## The nightly rate

Every model below was scored once on the primary holdout. The average gives each city equal weight. The area-typical rate is the training median for that neighbourhood and room type, with a fallback when fewer than 8 training listings support the group.

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

The lowest average error was $54.19 a night, from XGBoost. Ridge regression was $55.18, inside the 2% band (up to $55.28), and it is the simplest model in that band.

CatBoost averaged $54.81 and the equal blend averaged $53.69. Neither beat Ridge regression's $55.18 by the pre-set 5% margin, which would have required $52.42 or less. CatBoost was fit once, at the depth and learning rate already chosen for XGBoost, not on its own search. Neither CatBoost nor the blend is in the app.

Ridge is linear regression with a penalty that shrinks the weights. The penalty was chosen on the training folds, preferring the stronger penalty when it was within 1% of the best. Austin and San Diego use 100. Seattle uses 10. The model predicts the log of the rate and is converted back to dollars. Predictions are floored at zero and are not capped, so an expensive guess stays visible.

XGBoost is a little better in San Diego ($53.88 versus $57.27) and essentially tied in Austin and Seattle. Across the three cities the gap is about one dollar. That is inside the 2% rule, so the simpler model ships. The neural net was worse, at $63.26 on average.

Primary holdout, with a 95% interval from resampling the held-out cells 1,000 times.

| City | Listings | Cells | R² | R² interval | MAE | MAE interval | Median % error | Median % interval | Area-typical MAE | Error removed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin, TX | 2,101 | 31 | 0.511 | 0.449 to 0.665 | $72.13 | $56.38 to $85.13 | 25.9% | 22.7% to 28.1% | $112.75 | 36.0% |
| San Diego, CA | 817 | 23 | 0.601 | 0.517 to 0.673 | $57.27 | $45.40 to $75.88 | 21.5% | 19.2% to 24.5% | $91.62 | 37.5% |
| Seattle, WA | 730 | 8 | 0.522 | 0.489 to 0.592 | $36.13 | $31.43 to $40.24 | 19.9% | 17.3% to 22.4% | $49.40 | 26.9% |

## All three cities together

The city rows are the numbers to quote. Austin, San Diego, and Seattle are different markets: the usual nightly rate on this holdout is about $113, $92, and $49, and the holdouts are 2,101, 817, and 730 homes. One blended score would let Austin's prices, and its larger sample, speak for the other two cities.

Counting each city once is the overall the model was chosen on. Ridge's average miss is $55.18 a night, against $84.59 for the usual local rate, which is 34.8% less error. The average of the three R² values is 0.545, and the average of the three median percent errors is 22.5%. Resampling each city's holdout cells 1,000 times, then averaging the three cities, gives a 95% interval of R² 0.506 to 0.611, a miss of $48.86 to $63.45, a usual-rate miss of $77.07 to $100.80, a median percent error of 21.0% to 24.1%, and 31.6% to 40.4% less error.

Counting every holdout home once, instead of each city once, the miss is $61.60 against a usual rate of $95.34. That pool has R² 0.542, a median error of 23.6%, and 35.4% less error than the usual local rate. The same cell resampling, keeping each city's share of the homes, puts the R² between 0.480 and 0.659 and the miss between $50.26 and $71.34. The dollar error is higher than $55.18 because Austin has more of the holdout and more expensive misses.

| Summary | R² | Model miss | Usual local rate | Median % error | Less error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Each city once | 0.545 (0.506 to 0.611) | $55.18 ($48.86 to $63.45) | $84.59 ($77.07 to $100.80) | 22.5% (21.0% to 24.1%) | 34.8% (31.6% to 40.4%) |
| Every home once | 0.542 (0.480 to 0.659) | $61.60 ($50.26 to $71.34) | $95.34 ($79.54 to $109.27) | 23.6% (21.2% to 25.6%) | 35.4% (32.1% to 41.0%) |

The same specification on seeds 19, 29, and 37:

| City | Primary R² | Other seeds R² | Primary MAE | Other seeds MAE |
| --- | ---: | ---: | ---: | ---: |
| Austin, TX | 0.511 | 0.455, 0.396, 0.574 (spread 0.178) | $72.13 | $54.41, $64.01, $49.60 (spread $14.41) |
| San Diego, CA | 0.601 | 0.650, 0.650, 0.182 (spread 0.468) | $57.27 | $69.37, $74.25, $62.31 (spread $11.94) |
| Seattle, WA | 0.522 | 0.465, 0.544, 0.492 (spread 0.079) | $36.13 | $39.79, $36.18, $32.46 (spread $7.33) |

Seattle is stable. Austin's other seeds are a bit easier in dollars than the primary, so quoting $72.13 is not flattering. San Diego's R² falls to 0.182 on one of the three seeds. The model is useful against the area-typical rate, and it is not a number you should quote as if every map of the city would give 0.60.

The app's range is the 10th to 90th percentile of out-of-fold residuals, in five bands of the predicted price, fit on the training folds only. It covered 79.0% of Austin's holdout, 82.0% of San Diego's, and 81.8% of Seattle's. Median width was about $155, $115, and $83. The band was not widened to chase 80%.

On real holdout homes, putting one feature back to the training median moves the rate most for size and location. In Austin the largest average moves are bathrooms (about $52), distance to downtown (about $51), bedrooms (about $46), and guest capacity (about $39). In San Diego, longitude and bedrooms lead. In Seattle, bedrooms and the two distances lead. A separate check that flips one flag from no to yes, holding everything else typical, says a shared room or a private room lowers the rate and a waterfront flag raises it. Shared rooms are uncommon, so that swing is a coefficient, not a typical home. The app uses the first method: change one filled-in fact back to typical, and read the dollars.

## Booked nights

Same holdout, same 2% rule, simpler models only. The comparison is the average miss in nights.

| Model | Austin, TX | San Diego, CA | Seattle, WA | Average |
| --- | ---: | ---: | ---: | ---: |
| Area-typical nights | 72.0 | 76.5 | 72.8 | 73.8 |
| Linear regression | 68.9 | 70.2 | 68.7 | 69.3 |
| Ridge regression | 68.6 | 70.3 | 68.8 | 69.2 |
| Random forest | 67.9 | 70.5 | 68.7 | 69.0 |
| XGBoost | 69.0 | 69.9 | 68.7 | 69.2 |

The lowest average error was 69.0 nights, from Random forest. Linear regression was 69.3 nights, inside the 2% band (up to 70.4 nights), and it is the simplest model in that band.

A small neural net was also scored. Its average miss was about 69.0 nights, tied with the forest. It was not eligible to ship.

The bar, written down before this run and unchanged from the previous nights study, is an R² of at least 0.10 in every city and an average miss at least 10% lower than the area-typical nights. Both have to hold.

| City | R² | R² interval | MAE (nights) | MAE interval | Median % error | Area-typical MAE | Error removed | Cleared |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin, TX | 0.014 | -0.013 to 0.055 | 68.9 | 64.8 to 70.8 | 54.2% | 72.0 | 4.4% | No |
| San Diego, CA | 0.118 | 0.042 to 0.144 | 70.2 | 66.6 to 71.4 | 39.9% | 76.5 | 8.1% | No |
| Seattle, WA | 0.062 | -0.006 to 0.120 | 68.7 | 64.8 to 70.8 | 40.0% | 72.8 | 5.7% | No |

The nights model did not clear the pre-set bar in every city. The bar was an average miss at least 10% lower than the area-typical nights, and an R² of at least 0.10. The app labels nights as a rough estimate.

San Diego clears the R² half (0.118, and the interval stays above zero) and misses the 10% error cut (8.1%). Austin and Seattle miss both. The other seeds do not rescue it. Their R² values stay near the primary ones, and the dollar-style spread in nights is only about 1 to 3 nights.

| City | Primary R² | Other seeds R² | Primary MAE | Other seeds MAE |
| --- | ---: | ---: | ---: | ---: |
| Austin, TX | 0.014 | 0.032, 0.033, 0.057 (spread 0.025) | 68.9 | 68.5, 65.8, 66.6 (spread 2.8) |
| San Diego, CA | 0.118 | 0.100, 0.095, 0.060 (spread 0.039) | 70.2 | 71.2, 70.8, 73.1 (spread 2.3) |
| Seattle, WA | 0.062 | 0.073, 0.063, 0.056 (spread 0.016) | 68.7 | 70.5, 71.7, 71.9 (spread 1.4) |

That is an honest failure of the label more than of the model. A linear model, a forest, and XGBoost all sit within about a night of each other, and all of them sit close to the area median. The home's size and amenities do not tell you the review-based nights. The app still shows the number, labeled as a rough estimate, because hiding it would pretend the question was not asked.

Yearly revenue, predicted rate times predicted nights, has holdout R² 0.266, 0.276, and 0.161, and average misses of about $13,388, $14,093, and $9,107. The grade cuts are the 20th, 40th, 60th, and 80th percentiles of training revenue. On the holdout the letter matched the review-based revenue's letter 31.8% of the time in Austin, 31.0% in San Diego, and 30.4% in Seattle. Better than a coin flip among five letters, and not something to call accurate.

## What to say, and what not to say

Say that a simple model, which a person who knows linear regression can explain, matches the best tree model for the nightly rate within about a dollar on average, and beats the usual local rate by roughly a quarter to a third of the error. Say the test was geographic, frozen first, and checked on other seeds. Say the nights figure is rough and missed a bar that was not moved.

Do not say the model knows the booked nights. Do not say the range is an 80% promise. Austin's measured coverage is 79.0%. Do not quote San Diego's 0.601 as if the other seeds agreed. Do not say CatBoost or a blend is in the app. They were tried, and they did not clear the 5% bar.

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
