# Calendar snapshots and the booked-night rules

The nights model in this repo predicts a review-count proxy. Held-out R² on that proxy is 0.014 in Austin, 0.118 in San Diego, and 0.062 in Seattle. This note is the measurement plan for a label made of nights that were actually on a calendar. It is not a new score. The panel has to accumulate first.

## What is collected

Three cities: Austin, San Diego, Seattle. The ids are drawn once from the newest Inside Airbnb listings file for each city: Austin 2026-09-17, San Diego 2026-09-20, Seattle 2026-09-20. A row is eligible when `number_of_reviews_ltm` is at least 1 and `availability_365` is at least 1. That is a review in the twelve months before the file, and at least one open night on that scrape. The draw is seed 17. It is stratified by `neighbourhood_cleansed` and `room_type`, with largest-remainder allocation capped at the size of each stratum, then a seeded draw inside the stratum. The ids are committed in `data/calendar_sample.json`. The eligible rows are `data/calendar_pool/`. The daily job reads the sample. It does not redraw.

The ask is larger than 300 where the probe said a few calendars would come back empty, so the expected number of calendars is about 300. Austin and San Diego ask 309. Seattle asks 300. Listing ids are kept as text. The newer ids are longer than the 53 bits of a floating-point number, and rounding one changes the id. The public calendar for a rounded id comes back empty.

Each run appends one compact snapshot per city:

| Column | Meaning |
| --- | --- |
| `listing_id` | The public listing id |
| `date` | The night |
| `available` | 1 if the calendar shows the night open, 0 if it shows it closed |
| `min_nights` | The minimum stay posted on that night, when the calendar includes it |
| `price` | The nightly price, when the calendar includes it. Blank when it does not |
| `fetched_at` | When this row was observed, in UTC |

Files are gzipped CSV. A live run writes `calendars/daily/<city>/<YYYY-MM-DD>.csv.gz`. A fallback run writes `calendars/inside-airbnb/<city>/<snapshot-date>.csv.gz` and stamps `fetched_at` at noon UTC on the Inside Airbnb scrape date. A JSON report under `calendars/runs/` records the mode, the success count, and any stop.

The scheduled job commits those files to the `calendar-data` branch. Main keeps the collector, the sample, and this note. It does not keep the daily files.

## How the live fetch behaves

The client requests the public availability calendar for one listing, waits 2 seconds plus up to 1 second of jitter, then requests the next. Ordinary HTTP 429 and 5xx responses are retried up to three times, with waits of 5, 15, and 45 seconds. The user agent identifies CapScout and this repository. There is no login, no captcha solving, and no proxy.

A 401, a 403, an HTML page, or a challenge marker stops the live run. The client logs the status and the reason. It does not change identity and try again. Listings that return 404 are recorded as missing and the run continues. If the live endpoint produces no rows, the job falls back to Inside Airbnb's published calendar file for the same ids. Inside Airbnb's public index, checked while this was written, listed one calendar date per city. One file cannot show a night change from available to unavailable. The fallback stores that file so a later published date can be diffed. It does not invent bookings from it.

A city that already has a complete full-panel file for that UTC date is left as it is. A later manual run with a listing cap does not replace that file. The scheduled job copies the `calendar-data` files into its work directory first, so the second run of a day does not ask again.

The public calendar often returns a null price on both open and closed nights. The column is still stored. The rules below do not read it.

## Probe and the first collection

On 2026-11-19 the active pool was probed with 36 ids per city, drawn the same way as the panel, from the machine that prepared this change. GitHub Actions has not run the workflow. The schedule starts from the default branch after the workflow file is merged, so a hosted-runner result is still open. The public endpoint answered and the probe was not challenged.

| City | Returned a calendar | Empty calendar | Failed | Ids the daily job asks |
| --- | --- | --- | --- | --- |
| Austin | 35 of 36 | 1 | 0 | 309 |
| San Diego | 35 of 36 | 1 | 0 | 309 |
| Seattle | 36 of 36 | 0 | 0 | 300 |

309 times 35/36 is 300.4, and 300 times 36/36 is 300, which is the sizing rule: round the ask up so the probe rate yields about 300 calendars. At roughly 3 seconds a listing, the full pass is under an hour. That fits the 180-minute job, so the workflow does not split by city.

The same day, before this redraw, a panel of 300 ids from the 2025 study files was collected in full. Austin returned 80 calendars, San Diego 93, and Seattle 92, with no failures, from 16:00 to 16:40 UTC. Those files are the 2026-11-19 commit on `calendar-data`. They are the retired panel. The label command keeps ids in the current sample and leaves that day out of the rates.

## Where the rules come from

Inside Airbnb's data assumptions say the calendar does not differentiate a booked night from an unavailable night, and their own occupancy model therefore uses reviews. A single `calendar.csv` is a picture of the next year on one scrape date. Near nights are more often closed than far nights, and a closed night can be a reservation, a host block, a minimum-stay hole, or the far side of the host's availability window.

Wang, Livingston, and coauthors, in PLOS One (2024), "Enhancing our understanding of short-term rental activity," scrape daily and count a likely booking only when a night changes from available to unavailable. Nights that were never available are removed from the booking count. They also show that a weekly or fortnightly gap misses bookings and cancellations that open and close between scrapes. Inside Airbnb's release is about monthly, so it has that gap.

AirDNA's public description of occupancy is reserved nights divided by nights that were either reserved or available. Blocked nights stay out of the denominator. They say a private model uses length of stay, lead time, reviews, and other signals to separate blocks from reservations, and a 2024 note says advance-notice and preparation-time gaps were being counted as reservations and should not be. The model itself is not published. The rules here are the open substitute: length, lead time, and whether the night was ever seen open. They will disagree with a 16-signal model, and that disagreement is a limit, not a hidden input.

Airbnb's help text on the availability window says a host can open the calendar 3, 6, 9, 12, or 24 months out. The far tail of a 365-day scrape is often that setting. Advance notice and preparation time can also close a night or two around a stay without a guest occupying them.

## The rules

All of these are in `data/calendar_labels.py`. A run is a contiguous set of nights that change together between the same pair of snapshots.

1. **Booked.** The nights were available, then unavailable. The run is 1 to 13 nights long. The earlier snapshot still showed the check-in open, and that snapshot was inside 45 days of the check-in. The run is at least as long as the minimum stay posted on the check-in while it was open. The night is still unavailable on a snapshot within 2 days of the night itself.

2. **Owner block.** The run is 14 nights or longer, and it was already closed on a snapshot taken when the check-in was still 90 days out. Ninety days is the 3-month availability window, the shortest of the standard settings. The same label is applied when one snapshot closes at least half of the nights that were open, and at least 14 of them, and the two snapshots are at most 3 days apart. A guest does not reserve half a year overnight. A month of real bookings can remove that many nights, so this mass-block rule stays off when the gap is wider than 3 days.

3. **Closed window.** On the first snapshot that contains them, 21 or more nights are unavailable from some date through the last date that snapshot publishes. Those nights were never seen open. They are the shut end of the calendar, not demand.

4. **Censored.** The night was unavailable the first time it was seen, and it is neither a long far block nor a closed window. It may already have been booked. It may already have been blocked. The panel did not see the flip, so it is not called a booking.

5. **Uncertain.** The run is longer than 13 nights but was not observed closed while still 90 days out, or the gap between snapshots is wide enough that the check-in might have closed more than 45 days ahead. Both a long guest stay and a medium-lead booking land here. They are reported and left out of the rate.

6. **Too short.** The run is shorter than the minimum stay that was posted on the check-in. A one-night hole next to a three-night minimum cannot be a guest stay under the rule the host published. Advance-notice and preparation gaps often look like this. They are not bookings.

7. **Unbooked.** A snapshot within 2 days of the night still shows it available. The night was offered and was open at the end.

8. **Unresolved.** The night's date has passed, and no snapshot falls within 2 days of it. A monthly file leaves almost every night in this bin. That is intentional. Widening the bin to "whatever the last monthly file said" would call a month of unknown flips a measurement.

A night that goes unavailable and later goes available again, before the night, is unbooked if it ends open. The intermediate closure is a cancellation or a released block. Future nights keep a provisional label so the file shows what is on the books, and they stay out of the rate until the date arrives.

The booked-night rate for a listing is

`booked / (booked + unbooked)`

over nights whose date is on or before the last snapshot. Owner blocks, the closed window, censored nights, uncertain nights, too-short nights, and unresolved nights are counted in their own columns and left out of the denominator. An empty denominator is a missing rate, not zero.

## Limits

- The calendar still does not name the guest. An owner who blocks a weekend inside 45 days, on a listing whose minimum stay is 1 or 2, is counted as a booking. That is the main false positive.
- A real stay of 14 nights or more is an owner block when it was closed far out, and uncertain when it was closed near the date. Long stays are the main false negative. Calling them bookings would also call month-long host holds bookings.
- Stays booked more than 45 days ahead are uncertain, so the rate undercounts early bookers. San Diego in summer and Austin around a festival book earlier than a city weekend in the middle of the calendar. The rate is a short-lead fill rate of nights we saw still open, not a full occupancy.
- Nights already closed on the first snapshot are censored or blocks. The first weeks of the panel are a poor occupancy estimate because the near calendar is whatever was left.
- Preparation time of a night or two that flips in the same snapshot as the stay becomes part of that stay's length. We do not have the host's preparation setting, so we do not peel those nights off.
- Two back-to-back stays that close on the same day become one run. Visit counts would be low. Night counts are the target, so the merge is the smaller error.
- A snapshot gap longer than 2 days leaves those nights unresolved. GitHub's scheduler can skip a day. One missed day still resolves, because the slack is 2. A week-long outage does not.
- The panel is about 300 calendars per city, not every listing. A rate on this panel is not a city occupancy rate. The eligible pool is a few thousand homes per city, and the draw is stratified, so small neighbourhoods can be missing.
- Listings disappear, change id, or get blocked from the public calendar. The run report's missing count is that attrition.
- Inside Airbnb's monthly file, used only when the live endpoint is unusable, cannot support rule 1 for more than the nights within 2 days of the scrape. It is kept so the next file has something to diff against.

## When to retrain

Wait 26 weeks of daily snapshots.

A night that is already unavailable on day one is censored. A night that is first seen beyond 90 days becomes a resolved, uncensored outcome only when that date arrives. The first of those nights arrive about 13 weeks after the panel starts. Before that, the resolved nights are mostly whatever was still open in the near window, which is the leftover inventory and a downward-biased rate.

After that, offered nights show up about one per day. For a rate near 30%, the listing's own standard error is near 5 points once there are about 80 offered nights: 0.3 × 0.7 / 0.05² = 84. Twelve weeks supplies that many, if the calendar was mostly open when we first saw it. 13 + 12 = 25. The extra week makes 26, a half year, so the window is not only one holiday peak.

At 26 weeks, fit a model of this observed booked-night rate. Do not multiply it by 52/26 and call it a year. Score it once, on a fresh spatial holdout: a new seed, not seed 17 and not seed 31, with the held-out cells written down before the fit. Seed 17 is the rate study's primary holdout, and it is also the number used to draw this panel. The panel draw is not that holdout. The comparison is this label against a neighbourhood median on that holdout. The review-proxy R² values, 0.014 in Austin, 0.118 in San Diego, and 0.062 in Seattle, are not the bar for the new label. An annual night count, comparable to the review proxy's "nights per year," waits until 52 weeks of the panel exist.

```bash
python scripts/collect_calendars.py --out-dir calendars
python scripts/build_occupancy_labels.py --snapshots calendars --out reports/calendar_occupancy.csv
```

The second command reads every `csv.gz` under the snapshot directory and writes one row per listing.
