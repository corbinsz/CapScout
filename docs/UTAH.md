# Salt Lake City and Park City

The owner lives in Utah. CapScout does not score Salt Lake City or Park City today. This file is the path to add them without scraping Airbnb.

## What was checked

Inside Airbnb's public catalog, dated 14 September 2026 on https://insideairbnb.com/get-the-data/, lists the US cities this app already uses (Austin, Nashville, Denver, and the rest). It does not list Salt Lake City, Park City, or any other Utah region. A search of that page for "Salt Lake", "Park City", and "Utah" returns nothing.

Their US country archive is mentioned as a research file. It is not a free per-city download on that page, and it is not a license to scrape listings that are not published.

Inside Airbnb will add a new region by request: https://insideairbnb.com/data-requests/. A new region starts at $375. Commercial or non-mission requests (a product that helps people buy short-term rentals is in that bucket) are the lowest priority and have to be funded. The public files they do publish are Creative Commons Attribution 4.0. A commissioned Utah file would be too, if they agree to publish it. Email data@insideairbnb.com or use the form on the data-request page. Ask for Salt Lake City and Park City as two regions, listings plus reviews, and say the use is a commercial investment screener so the quote is the commercial one.

## A licensed feed that already covers Utah

AirROI sells Airbnb market data, including Utah, through an API: https://www.airroi.com/api.

- Key header: `X-API-KEY`
- Market lookup is about $0.01 a call (`GET /markets/search`)
- Market occupancy is about $0.05 to $0.10 a call (`POST /markets/metrics/occupancy`)
- A listing search for a market is about $0.25 to $0.50 a call (`POST /listings/search/market`)
- Credits start with a $10 deposit. There is no monthly minimum.
- A bulk file is a separate purchase. Their non-commercial price guide starts at $300 to $1,000 per market depending on the tier. A commercial file needs a quote: https://www.airroi.com/data-portal/request-data

That is enough to add a market summary. It is not a drop-in replacement for the Inside Airbnb training file, which has one row per listing with coordinates, bedrooms, and reviews. Training a CapScout model needs that row-level file. The practical order is:

1. Call `GET /markets/search` with Salt Lake City and with Park City and confirm the locality names AirROI uses.
2. Buy or request a listing extract (coordinates, bedrooms, baths, property type, reviews in the last year, minimum nights, price).
3. Add two entries to `backend/cities.py` with tax, insurance, permit cost, and the city center.
4. Save the file as `backend/data/cities/<key>/listings.csv.gz` using the same columns the cleaner already reads.
5. Run `python -m models.benchmark --cities salt-lake-city` (or the occupancy retrain) and only then show the city in `src/utils/markets.js`.

Do not point the app at Airbnb's website to fill the gap. RentCast can list homes for sale in Utah once a key is set, but it does not replace the short-term rental history the model is trained on.

## What not to do

Do not invent scores for Utah from the Denver model. The mountains and the rules are different, and a Denver score on a Park City house would look precise and be wrong.
