# Explain it yourself

One page for an interview. Say it in this order.

**The question.** CapScout estimates two things for a home in Austin, San Diego, or Seattle: the nightly rate it could advertise on Airbnb, and a rough guess of nights booked in a year. The app multiplies them and gives that yearly figure a letter grade inside the city. It runs on my laptop. It is not a website.

**The data.** I used public Inside Airbnb snapshots: Austin from 16 September 2025, San Diego and Seattle from 16 March 2025. I parsed prices like `$1,234.00`, dropped missing and junk prices outside $25 to $2,000, dropped homes with no review in the last year or blocked every night, dropped duplicate ads, and kept only homes within 60 km of downtown. Austin went from 10,533 rows to 7,720. San Diego went from 12,991 to 8,387. Seattle went from 6,772 to 4,610. I joined census tract income, rent, and home value, checked that almost every home landed in a tract, and then left census out because it did not improve the training error enough to be worth the extra columns.

**The test, before the model.** I cut each city into 3 km cells and held out 20% of the cells with a seed I wrote down first (17). Every home in those cells is held out. I also removed, from training only, any host who has a held-out home and any training home within 1 km. So the model cannot learn the house next door or another listing from the same host. I did not touch that holdout while choosing features or settings. Three other seeds (19, 29, 37) repeat the same choice so I can see if one map of the city was a fluke.

**The models.** I compared the usual rate for that neighbourhood and room type, linear regression, ridge regression, a random forest, and XGBoost, all on the log of the nightly rate. A small neural net was a comparison only. CatBoost, which I have not used and do not want to defend, was fit once. An equal blend of CatBoost and XGBoost was scored once. The rule was: ship the simplest model within 2% of the best error. CatBoost or the blend would ship only with a 5% win. Neither got it.

**Why this model.** XGBoost had the lowest average miss, $54.19 a night. Ridge was $55.18. The 2% cutoff was $55.28, so ridge wins because it is simpler. Ridge is linear regression with a penalty that keeps the weights small. The things that move the rate are size, distance to downtown, and room type. In Austin the average miss is about $72, in San Diego about $57, in Seattle about $36. That is about 36%, 38%, and 27% less error than the usual local rate. A typical home is off by about 20% to 26%. The dollar error is larger because a few expensive homes miss by a lot.

**The catch.** San Diego's R² is 0.601 on the main holdout and 0.182 on one of the other seeds. I say the main number and the spread. I do not average them into a nicer figure. The range in the app covered 79.0% of Austin's holdout, 82.0% of San Diego's, and 81.8% of Seattle's. I quote those, and I did not widen the band after seeing them.

**Nights.** The label is reviews in the last year, divided by 0.50, times about 3 nights, capped at 255.5. It is not a count of bookings. The model is not allowed to see the review count, the price, or how many nights are blocked. Linear regression was within 2% of the best nights model, so that is what ships. It missed a bar I set first: every city needed R² of at least 0.10 and an error at least 10% below the usual nights for the area. Austin's R² is 0.014. The app calls nights a rough estimate. I do not call it a forecast.

**If they ask what I would not claim.** I would not claim profit, a legal short-term rental, or what a guest actually paid. The grade matched the review-based revenue's grade about 30% of the time. That is a rank, not a verdict.
