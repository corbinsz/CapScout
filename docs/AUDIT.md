# Audit notes

These are the problems found before the changes in this branch. The fixes are in CHANGES.md.

## Bugs

- RentCast and the Realtor client cached responses with `config.DATA_PROCESSED_DIR`, which did not exist. The first cache write crashed.
- “Top deals” scored only the 100 cheapest listings, then sorted those. Expensive homes never entered the ranking.
- Sorting by price applied the confidence filter after the page was cut, so pages came back short.
- Half bathrooms were stored as whole numbers.
- A for-sale id that was not a Redfin id 404’d, because only `redfin-` ids were looked up in the sale file.
- Search looked at Redfin only.
- The score cache on disk never expired.
- Property links omitted the city, so a shared link opened in whatever city the browser had saved.
- The UI said the model had no holdout and showed R-squared near 0.97. That R-squared was an in-sample fit on area-median targets. On a real holdout the old nightly-rate error was about $59 to $228 depending on the city, and R-squared on raw price was negative.
- `settings` could write API keys into `.env` with no login.
- City names were joined onto a filesystem path. A value like `../../` could escape the data directory.
- CSV upload had no size cap.
- Flask started with `debug=True` whenever you ran `python app.py`.
- CORS allowed every origin.
- The font was named Inter and was never loaded.

## Data

- The for-sale path preferred Redfin’s unofficial map endpoint. That is a poor fit for a paid product.
- Inside Airbnb’s public files are CC BY 4.0. Credit is required. The project also asks users not to republish the raw data and not to scrape the site. Occupancy from the calendar confuses blocks with bookings.
- Hawaii is a whole state centered on Honolulu, so “distance to downtown” is a weak feature there. Called out rather than silently “fixed.”

## Security

- No accounts, no rate limit, no security headers.
- Secrets were not found committed in the tree. The settings form was the way a secret could have been posted to the server.

## Performance

- Scoring walked homes one at a time. The list now scores the filtered set in one model call and caches it for six hours.
- Map and chart libraries are in separate bundles, so the home page does not download them.

## What was not a code bug

Salt Lake City is not a supported market. The owner lives in Utah. The app should not pretend otherwise, and it should not treat Denver as a home base.
