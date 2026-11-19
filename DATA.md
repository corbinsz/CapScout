# Data

The current scored model is the simple spatial study in `reports/FINDINGS.md`: ridge regression for the nightly rate, and a linear model for a rough nights estimate. Raw Inside Airbnb files are not in git. Download them with `python -m data.download austin san-diego seattle`. Older history files, if you want them, come from `python -m data.history` and are gitignored the same way.

Every source below can be downloaded by a script and reused for this study. Derived columns (distances, counts, tract attributes joined to listings) are what a model can see. The national Census files and the raw Overpass responses are caches and are not committed. The city extracts under `data/external/` are small and stay in the repo.

No Redfin, Zillow, AirDNA, or other source whose terms forbid automated collection is downloaded or used.

## Inside Airbnb listings

- **What:** one snapshot per city, `data/raw/cities/<city>/listings.csv.gz`.
- **Markets in the model:** Austin (2025-06-13), San Diego (2025-03-16), Seattle (2025-03-16).
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0). Credit Inside Airbnb. These files are filtered (price window, required bedrooms and coordinates) and are not a verbatim republication of the full dump.
- **Source:** [insideairbnb.com/get-the-data](https://insideairbnb.com/get-the-data/) and [insideairbnb.com/data-policies](https://insideairbnb.com/data-policies/). Inside Airbnb is not Airbnb and is not affiliated with this project.
- **Join:** there is no join. The row is the listing. `python -m data.download austin san-diego seattle` fetches the three snapshots the metrics were computed on. Those dates are pinned in `data/cities.py`. The files are gitignored, so they are not in a fresh clone until that command runs.

Other snapshots are on disk and were screened. They are not in the model. Hawaii is several islands in one file. Asheville has almost no bedrooms. Las Vegas, Fort Lauderdale, Nashville, and Washington are missing a nightly price on roughly 20–30% of rows. Boston, Denver, Dallas, Fort Worth, and New Orleans are missing a price on about 9–20%. A fourth market with a large unpriced slice would not make the spatial test more honest.

### Earlier snapshots of the same three cities

Inside Airbnb's public page keeps about the last 12 months. On the day this round was frozen, every file still hosted for San Diego and Seattle before their study dates was one snapshot, 2025-02-18. Austin still had a monthly file from 2024-04-21 through 2025-05-11. Files dated after the study snapshot (Austin 2025-06-13, San Diego and Seattle 2025-03-16) were left alone: a later scrape is a price from after the label. The files belong under `data/raw/cities/<city>/history/<date>.csv.gz`, CC BY 4.0, credit Inside Airbnb. `python -m data.history` downloads any that are missing. They are gitignored, same as the current snapshots, and the current model does not use them.

A listing that is already in the study file is not repeated. The newest earlier snapshot is kept for ids the study file does not have. Those rows can enter training only. They are dropped when they share a host with a held-out listing, sit in a held-out cell, or fall inside the 1 km buffer. Whether they cleared the error bar is in the findings, not here.

### Not used from Inside Airbnb

- **Calendar and availability.** Inside Airbnb says a night marked unavailable is not necessarily booked. `availability_365` is not a feature and it is not the nights label. A commercial calendar model that tries to separate booked nights from blocked nights is not in this data.
- **The review-based nights estimate is a separate label, not a price feature.** `number_of_reviews_ltm`, divided by a 50% review rate, times a stated 3-night stay (or a longer minimum up to 90 nights), capped at 70% of the year, is the occupancy study's target. The price model does not train on it, and the nights model does not train on the review count, the minimum stay, or the price. Listings with no reviews in the last 12 months are left out of that study. The assumptions are in `docs/OCCUPANCY.md`.
- **`review_scores_value`.** Guests score value by comparing the stay to the price. The other review scores (cleanliness, location, accuracy, check-in, communication) are in the model. The value score is not.
- **The review text file.** Sentiment over the raw reviews is a separate download and was not fit. The aspect scores on the listing file were used instead.
- **Photos.** `picture_url` points at a host's image. The photograph is not part of the CC BY listing table and was not downloaded.
- **Neighbourhood medians of price, nearest-listing prices, and a 5 km Gaussian smooth of training prices,** except as experiments fit inside the training fold. None of them cleared the bar. See the findings. They are not in the scored model.
- **Listing text.** The title, description, and neighbourhood blurb were turned into a fixed lexicon and a fold-fit TF-IDF. Neither cleared the bar. The text is still built by the feature code so the negative result can be rerun. It is not in the scored model.

## U.S. Census Bureau, ACS 2024 5-year

- **What:** tract median household income (table B19013), median gross rent (B25064), median home value (B25077), and total population (B01003). Density is population divided by tract land area.
- **License:** U.S. government work, public domain.
- **Source files, not the API.** The Census data API now requires a key. The same published estimates are on the table-based summary files, which a script can download with no account:

  `https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/data/5YRData/acsdt5y2024-<table>.dat`

  Tract rows are the lines whose geography id starts with `1400000US`. The 11-digit tract id is the last 11 characters.

- **Boundaries:** TIGER/Line 2024 census tracts, public domain.

  `https://www2.census.gov/geo/tiger/TIGER2024/TRACT/tl_2024_<statefp>_tract.zip`

  Land area is the `ALAND` field, in square meters.

- **Join:** point-in-polygon. The listing coordinate is tested against tract polygons whose bounding box meets the city, padded by 0.12 degrees so the edge of the metro is included. Polygons are simplified by about 15 meters before they are stored, which is smaller than the roughly 150 meter coordinate jitter Inside Airbnb applies. If a point misses every polygon but is within 2 km of a tract centroid, that tract is used. If an estimate is missing (ACS uses negative sentinels; a handful of tracts have no median rent or home value), the city median of the tracts that do have the value is filled in. That fill does not use listing prices. On the cleaned rows in this study, 99.8% of Austin listings, 100.0% of San Diego listings, and 99.9% of Seattle listings hit a polygon directly. Census columns were joined and then left out of the scored model, because they did not cut training-fold error by the pre-set 2%.

| City | Tracts kept | Tracts with income | Median income | Median rent | Median home value | People per km² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Austin | 448 | 444 | $103,312 | $1,802 | $481,400 | 1,240 |
| San Diego | 701 | 693 | $110,065 | $2,354 | $817,800 | 2,426 |
| Seattle | 452 | 447 | $129,181 | $2,170 | $852,900 | 2,265 |

The medians above are medians of tracts, not of listings.

## OpenStreetMap

- **What:** coordinates of food places (restaurant, cafe, fast food), transit (bus stops, rail stations and halts, tram stops, subway stations), parks and gardens, beaches, and attractions (attraction, museum, viewpoint). Names are dropped. The model sees distance to the nearest object, capped at 20 km, and the count within 1 km.
- **License:** Open Database License (ODbL). © OpenStreetMap contributors. The cached points are a filtered excerpt for these three bounding boxes. If you redistribute that excerpt, keep this attribution and the ODbL. Derived distances and counts are what the study publishes in the metrics.
- **Source:** a public Overpass endpoint, queried by `python -m data.external`. The endpoint used successfully for this run was `https://maps.mail.ru/osm/tools/overpass/api/interpreter`, with `https://overpass.osm.ch/api/interpreter` as the fallback. A city-sized query that comes back empty is retried as four quadrants. That happened for restaurants; the cached counts are the result after that retry.
- **Join:** each listing coordinate is projected locally and compared with a tree of the POI coordinates. Parks and beaches use the object's geometry nodes, so the distance is toward the edge of a large park rather than its center. Food, transit, and attractions use the object's center.

| City | Food | Transit | Park nodes | Beach nodes | Attractions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Austin | 3,237 | 2,390 | 34,360 | 212 | 110 |
| San Diego | 4,227 | 4,747 | 65,777 | 2,706 | 413 |
| Seattle | 5,259 | 2,200 | 36,550 | 2,771 | 403 |

Park "points" are geometry nodes, so the count is not a count of parks. Distance to a park is short in most of these cities. The ablation found that this block did not reduce error. It is not in the scored model.

## Airport and downtown

Distance to downtown uses the city reference point in `data/cities.py`. Distance to the airport uses FAA reference points typed in the same file (AUS, SAN, SEA), rounded to about 100 meters. Those are not a download. They do not depend on anyone's price.

## What the scored model actually uses

The nightly-rate model is ridge regression on the log of the advertised rate. Its columns are size (bedrooms, bathrooms, beds, guests), room type, house versus condo, latitude and longitude, distance to downtown and to the airport, a short amenity list, review score, review count, and superhost. The nights model uses the same home and amenity columns and does not use reviews, price, minimum stay, or availability. Census tract attributes are joined and documented above, and they are not in the scored models. OpenStreetMap columns were measured in an earlier round and were not added again. Nearby ads shown in the app are not model inputs.
