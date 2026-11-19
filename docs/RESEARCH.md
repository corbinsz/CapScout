> Background from an earlier round. The current measurement is in [reports/FINDINGS.md](../reports/FINDINGS.md).

# What this round read, and what it did with it

CapScout predicts an Airbnb listing's advertised nightly rate in Austin, San Diego, and Seattle. This note is the reading behind the round that froze a third holdout and tested more data, the high-price tail, text embeddings, and a price-band interval. It is a personal project. The citations are the public papers and data notes I actually used. The measurement is in [reports/FINDINGS.md](../reports/FINDINGS.md).

## The question, unchanged

The frame is still a hedonic regression. Rosen (1974) is the idea: a price is a bundle of attributes. Gibbs, Guttentag, Gretzel, Morton, and Goodwill (2018) and Wang and Nicolau (2017) fit that bundle to Airbnb rates, on a log price, city by city. CapScout already does that. This round did not reopen the target or the city-by-city blend.

The split is still the one Roberts and colleagues (2017) argue for when the rows are blocked in space. A random row split trains on the listing next door. Seed 7 and seed 11 had both been read, so this round froze seed 23 first, from the cells in neither of those, and did not use it to choose anything.

## What the public pricing write-ups add

Student and Kaggle write-ups on Airbnb rates, including the Duke CS 671 note and the public CatBoost/LightGBM stacking repos, tend to do four things this study had not measured: more rows, a tuned or stacked booster, a text embedding, and a correction for the expensive tail. Several of them also report a random holdout. I kept the spatial holdout and tested the ideas that can be explained without it.

**More snapshots, only from before the label.** Inside Airbnb's public page keeps about the last twelve months and points older files to a data request. Every file still hosted, and dated before the study snapshot, was used: Austin monthly from April 2024 through May 2025, and one February 2025 file each for San Diego and Seattle. Later scrapes, including the 2026 files on the public page, were not used. A price from after the label is not training data for that label. Listings already in the study file were not repeated. The extra rows missed a 1% development bar. Seattle moved by about two dollars. Austin, which had the year of extra listings, did not.

**The tail.** A log model, converted back with expm1, aims at something like a conditional median. Duan (1983) smearing multiplies by the mean of exp(residual) to aim at the mean instead. That is the right correction if the loss is squared error. The loss here is absolute error, which wants the median, and smearing made development MAE worse. An isotonic map from predicted dollars to actual dollars, fit on out-of-fold predictions, was worse again. The map does not travel from one block of the city to another. Up-weighting listings above the median, on a fixed formula that gives the 90th percentile three times the weight of the median, also made absolute error worse. It spends the fit on the tail the metric does not reward. I did not keep any of the three.

**Another tree, a small grid, a weighted blend.** v2 had already found LightGBM a bit worse than XGBoost alone. Adding it as a third equal-weight model, on top of the CatBoost and XGBoost blend, was a bit worse than the blend. A grid of three settings (learning rate 0.03, learning rate 0.08, minimum child weight 20) was scored on two development seeds and would have been kept only if the other three seeds also cleared 1%. The best of the three was about 0.2% better on the screen and was not confirmed. I did not run Optuna. An adaptive search on the same five seeds would have been another chance to fit those seeds. One public write-up of a similar contest found that an untuned booster on careful features beat a long tuning run. That matched this measurement.

A non-negative least-squares blend, weights fit on seeds 0 and 1 only, put most of Austin on CatBoost, most of San Diego on XGBoost, and all of Seattle on CatBoost. On seeds 2–4 it was about 0.2% better than equal weights. The bar was 1%. Equal weights stayed.

**Text, one more time.** Kalehbasti, Nikolenko, and Rezaei (2019) add review sentiment and report a high R² on a random-style split. v3 had already tried a lexicon and an eight-component TF-IDF on the ad, and both missed the bar. This round tried a small static sentence embedding, MinishLab's potion-base-8M, compressed to the same eight components with a PCA fit inside the training fold. The embedding model was not trained on these listings. Development MAE was flat. The ad is mostly size, room type, and amenities the structured columns already have. Photos were still not downloaded. The picture URL is not part of the CC BY table.

**Comps, one more time.** Matched on room type and bedroom bin, within 15 km, at least eight matches, own price removed. v2's neighbour median and v3's 5 km kernel had failed under the buffer. The matched version failed too, and by more. A held-out cell is not priced by the training listings 15 km away.

## The interval

v3's band scales with the predicted price and still missed cheap Austin listings. Mondrian conformal prediction fits a separate quantile in each region of the features. Boström and Johansson (2020) is the regression version I used: bin the calibration residuals, and give each bin its own interval. Hjort (2022) is the housing note, one interval per district on Norwegian sales. I used quintiles of the predicted price, which is the region where v3's coverage failed, and an asymmetric residual so the band can be longer above the estimate than below it. A band with fewer than 40 calibration residuals uses the city-wide quantile.

The rule, written down first, keeps the scaled band unless a challenger raises the worst city's worst quintile by at least 3 points, without dropping the worst city by more than a point, without getting more than 35% wider, and without leaving the point estimate outside the band. The Mondrian band did that on the development folds: the worst quintile went from 64.3% to 71.7%, and the median width went from about $123 to about $118. The asymmetric band without the quintiles cleared the 3 points and was wider, so the quintile version won. The test was not used to switch.

## What I did not reopen

Depth, monotone constraints, a pooled three-city model, the calendar, `review_scores_value`, and OpenStreetMap. Each of those already had a result. Photos still were not downloaded.

## Sources

- Boström, H., and Johansson, U. (2020). Mondrian conformal regressors. *Proceedings of the Ninth Symposium on Conformal and Probabilistic Prediction and Applications*, PMLR 128, 1–20.
- Duan, N. (1983). Smearing estimate: a nonparametric retransformation method. *Journal of the American Statistical Association*, 78(383), 605–610.
- Gibbs, C., Guttentag, D., Gretzel, U., Morton, J., and Goodwill, A. (2018). Pricing in the sharing economy: a hedonic pricing model applied to Airbnb listings. *Journal of Travel & Tourism Marketing*, 35(1), 46–56. https://doi.org/10.1080/10548408.2017.1308292
- Hjort, A. (2022). House price prediction with confidence: empirical results from the Norwegian market. *Proceedings of the Eleventh Symposium on Conformal and Probabilistic Prediction with Applications*, PMLR 179, 313–315.
- Inside Airbnb. Get the data, and the note that the public archive is about the last 12 months. https://insideairbnb.com/get-the-data/
- Kalehbasti, P. R., Nikolenko, L., and Rezaei, H. (2019). Airbnb price prediction using machine learning and sentiment analysis. arXiv:1907.12665.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., and Wasserman, L. (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association*, 113(523), 1094–1111.
- MinishLab. potion-base-8M, a static sentence embedding. https://huggingface.co/minishlab/potion-base-8M
- Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., and Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40, 913–929. https://doi.org/10.1111/ecog.02881
- Romano, Y., Patterson, E., and Candès, E. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems*, 32.
- Rosen, S. (1974). Hedonic prices and implicit markets: product differentiation in pure competition. *Journal of Political Economy*, 82(1), 34–55.
- Wang, D., and Nicolau, J. L. (2017). Price determinants of sharing economy based accommodation rental: a study of listings from 33 cities on Airbnb.com. *International Journal of Hospitality Management*, 62, 120–131.
- Wang, Y. (2024). In-class Kaggle competition writeup, Airbnb price prediction. Duke University, CS 671. The public note is the feature-engineering comparison, not a method I copied.
