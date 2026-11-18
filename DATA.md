# Data

Every source below can be downloaded by a script and reused for this study. Derived columns (distances, counts, tract attributes joined to listings) are what the model sees. The national Census files and the raw Overpass responses are caches and are not committed. The city extracts under `data/external/` are.

No Redfin, Zillow, AirDNA, or other source whose terms forbid automated collection is downloaded or used.

## Inside Airbnb listings

- **What:** one snapshot per city, `data/raw/cities/<city>/listings.csv.gz`.
- **Markets in the model:** Austin (2025-06-13), San Diego (2025-03-16), Seattle (2025-03-16).
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0). Credit Inside Airbnb. These files are filtered (price window, required bedrooms and coordinates) and are not a verbatim republication of the full dump.
- **Source:** [insideairbnb.com/get-the-data](https://insideairbnb.com/get-the-data/) and [insideairbnb.com/data-policies](https://insideairbnb.com/data-policies/). Inside Airbnb is not Airbnb and is not affiliated with this project.
- **Join:** there is no join. The row is the listing. `python -m data.download` can refresh a snapshot; the committed files are the ones the metrics were computed on.

Other snapshots are on disk and were screened. They are not in the model. Hawaii is several islands in one file. Asheville has almost no bedrooms. Las Vegas, Fort Lauderdale, Nashville, and Washington are missing a nightly price on roughly 20–30% of rows. Boston, Denver, Dallas, Fort Worth, and New Orleans are missing a price on about 9–20%. A fourth market with a large unpriced slice would not make the spatial test more honest.

### Not used from Inside Airbnb

- **Calendar.** Inside Airbnb says a night marked unavailable is not necessarily booked. Treating it as occupancy would invent a label, and availability also depends on the price being predicted, so it would be a leak. It is not a feature.
- **Neighbourhood medians of price, and nearest-listing prices,** except as an experiment that was fit inside the training fold and did not clear the bar. See the findings. They are not in the scored model.

## U.S. Census Bureau, ACS 2024 5-year

- **What:** tract median household income (table B19013), median gross rent (B25064), median home value (B25077), and total population (B01003). Density is population divided by tract land area.
- **License:** U.S. government work, public domain.
- **Source files, not the API.** The Census data API now requires a key. The same published estimates are on the table-based summary files, which a script can download with no account:

  `https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/data/5YRData/acsdt5y2024-<table>.dat`

  Tract rows are the lines whose geography id starts with `1400000US`. The 11-digit tract id is the last 11 characters.

- **Boundaries:** TIGER/Line 2024 census tracts, public domain.

  `https://www2.census.gov/geo/tiger/TIGER2024/TRACT/tl_2024_<statefp>_tract.zip`

  Land area is the `ALAND` field, in square meters.

- **Join:** point-in-polygon. The listing coordinate is tested against tract polygons whose bounding box meets the city, padded by 0.12 degrees so the edge of the metro is included. Polygons are simplified by about 15 meters before they are stored, which is smaller than the roughly 150 meter coordinate jitter Inside Airbnb applies. If a point misses every polygon but is within 2 km of a tract centroid, that tract is used (about 0.1–0.2% of rows). If an estimate is missing (ACS uses negative sentinels; a handful of tracts have no median rent or home value), the city median of the tracts that do have the value is filled in. That fill does not use listing prices. In this run, 99.8% of Austin rows, 99.9% of San Diego rows, and 99.9% of Seattle rows hit a polygon directly.

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

Census tract attributes, plus the listing's own size, coordinates, distances, and ad-level fields (amenities, reviews, host). OpenStreetMap columns were built and measured and left out. The findings record the comparison.
