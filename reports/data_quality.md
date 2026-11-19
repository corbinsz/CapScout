# Data quality

Counts are sequential. Each rule runs on the rows the previous rule kept.
The dollar cuts and the 60 km cut are fixed. They are not percentiles of the file.

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

## Sources

- **Austin, TX** snapshot 2025-09-16, CC BY 4.0, credit Inside Airbnb. `https://data.insideairbnb.com/united-states/tx/austin/2025-09-16/data/listings.csv.gz`
- **San Diego, CA** snapshot 2025-03-16, CC BY 4.0, credit Inside Airbnb. `https://data.insideairbnb.com/united-states/ca/san-diego/2025-03-16/data/listings.csv.gz`
- **Seattle, WA** snapshot 2025-03-16, CC BY 4.0, credit Inside Airbnb. `https://data.insideairbnb.com/united-states/wa/seattle/2025-03-16/data/listings.csv.gz`

Inside Airbnb's catalog is https://insideairbnb.com/get-the-data/ and the license is https://creativecommons.org/licenses/by/4.0/.

## Notes

- **Austin, TX.** 10,517 prices contained a dollar sign and 314 contained a thousands comma. At the bathroom step, 0 bathrooms were read from the text (0 of those were half-baths set to 0.5). The farthest kept listing is 30.0 km from downtown. Still missing at the end: 3 bathrooms, 2 beds, 0 amenity lists, 0 review scores.
- **San Diego, CA.** 11,704 prices contained a dollar sign and 369 contained a thousands comma. At the bathroom step, 0 bathrooms were read from the text (0 of those were half-baths set to 0.5). The farthest kept listing is 42.9 km from downtown. Still missing at the end: 0 bathrooms, 23 beds, 0 amenity lists, 0 review scores.
- **Seattle, WA.** 6,008 prices contained a dollar sign and 20 contained a thousands comma. At the bathroom step, 0 bathrooms were read from the text (0 of those were half-baths set to 0.5). The farthest kept listing is 14.5 km from downtown. Still missing at the end: 0 bathrooms, 3 beds, 0 amenity lists, 0 review scores.

## Census join

ACS 2024 5-year tables B19013, B25064, B25077, and B01003, joined to TIGER/Line 2024 tracts. Public domain. A missing tract statistic is the median of the city's tracts, not a price.

- **Austin, TX.** 99.8% of listings fell in a tract polygon, 0.2% used a tract within 2 km, 0.0% used the city median. 7.0% had a missing tract statistic filled with the city median.
- **San Diego, CA.** 100.0% of listings fell in a tract polygon, 0.0% used a tract within 2 km, 0.0% used the city median. 0.6% had a missing tract statistic filled with the city median.
- **Seattle, WA.** 99.9% of listings fell in a tract polygon, 0.1% used a tract within 2 km, 0.0% used the city median. 1.4% had a missing tract statistic filled with the city median.
