# Calendar snapshots

Daily availability snapshots for the fixed CapScout panel: 300 listings in Austin, 300 in San Diego, and 300 in Seattle. The collector, the sample ids, and the booked-night rules live with the study code. This branch holds the files.

Each city-day file is a gzipped CSV with `listing_id`, `date`, `available`, `min_nights`, `price`, and `fetched_at`. A JSON file beside it records how many listings returned a calendar. `calendars/runs/` is the run report.

The 2026-11-19 collection used the public calendar. Austin returned 80 listings, San Diego 93, and Seattle 92. The other ids in the panel came back with an empty calendar and were counted missing. None failed. The run was not challenged. Price was blank on every stored night.
