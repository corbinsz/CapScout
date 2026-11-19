# What this round read, and what it did with it

CapScout predicts an Airbnb listing's advertised nightly rate in Austin, San Diego, and Seattle. This note is the reading behind the round that added host and review-aspect features, tried the listing text, and changed the prediction interval. It is a personal project. The citations are the public papers and data notes I actually used. The measurement is in [reports/FINDINGS.md](../reports/FINDINGS.md).

## The question the literature is answering

Most of the public work is a hedonic regression. Rosen (1974) is the frame: a price is a bundle of attributes, and the regression is trying to recover what each attribute is worth. Gibbs, Guttentag, Gretzel, Morton, and Goodwill (2018) fit that model to 15,716 Airbnb listings in five Canadian cities. Physical characteristics, location, and host characteristics moved the published rate. More reviews were associated with a lower rate, not a higher one. Wang and Nicolau (2017) found the same kinds of attributes across 33 cities: capacity, a superhost badge, amenities, and the rating. Both papers use a log price. CapScout already did. This round did not change the target. Prices here are at least $25, so log and log1p are almost the same transformation, and every reported metric is converted back to dollars.

The practical warning in that literature is the split. A random row split trains on the listing next door, and advertised rates are smooth in space, so the score flatters the model. Roberts and colleagues (2017) go through this for ecological data: when the rows are blocked in space, random cross-validation is the wrong estimate of error on a new block. CapScout's 1 km buffer and grid cells are that argument applied to three cities. I did not replace it with a random split to chase a higher R².

## What I kept

**Structured attributes, per city.** Gibbs et al. and Wang and Nicolau both treat size, room type, amenities, and host attributes as the hedonic bundle, and they fit cities separately or with city effects because the slopes are not the same market. v2 had already kept size, location, census, amenities, and a coarse host block. This round added the rest of that bundle that is actually on the Inside Airbnb file: beds, a shared bathroom, host tenure, response rate, acceptance rate, response time, the separate review scores, and a few more amenities. The detail block cut development MAE by 2.0%, from $59.85 to $58.62, which was the pre-set bar. It stayed.

**Log price, dollar metrics.** The papers report coefficients on log price. A hiring manager asking "how many dollars are you off" needs MAE, RMSE, and MAPE on the dollar scale. Both are in the report. R² on the log price is not.

**A spatial holdout, and a second one once the first had been read.** v2 froze seed 7 because an earlier seed had been published. Seed 7 was then published too. This round did not use it to choose features. Seed 11 was frozen first, from the development cells only, and scored once. That is the same idea as Roberts et al., applied to the fact that a published test is no longer blind.

**SHAP as a sentence, not a feature-selection tool.** Lundberg and Lee (2017) is the attribution method. Boucher's Quebec hedonic note uses the same idea: the tree model and the regression should be telling the same story about size and location. SHAP here explains a prediction. It was not used to drop columns. The app turns the largest contributions into a sentence.

**An interval that can change width.** v2's band was split conformal on the absolute log residual (Lei, G'Sell, Rinaldo, Tibshirani, and Wasserman, 2018, is the regression version; the finite-sample quantile is the same idea as inductive conformal regression in Papadopoulos, Proedrou, Vovk, and Gammerman, 2002). One radius in log space makes the dollar band wider for expensive homes, but only in proportion to the price. Austin's expensive listings were still under-covered. Romano, Patterson, and Candès (2019) combine quantile regression with a conformal correction so the width can depend on the features. A simpler cousin scales the log residual by a function of the predicted price. Both were fit. The scaled band won on the development folds: the expensive fifth of predictions went from 69.7% coverage to 77.4%, and the median width stayed about $124. The quantile version was wider and did not win that comparison. On the locked test the scaled band covered 73.2% of Austin, up from 66.7%, and still short of 80%. The test was not used to switch.

## What I did not keep

**Listing text, in the two forms I was willing to explain.** Kalehbasti, Nikolenko, and Rezaei (2019) add a sentiment score from the reviews and report an R² of 0.69 on log price for New York, with a random-style holdout and a support-vector model. Almeida, Nunes, and Pinto Machado (2025) find that review sentiment moves the price more than the star rating, in a spatial lag model for Porto. The signal they are pointing at is real in their setups. Two things kept me from copying the method.

The review corpus is a separate Inside Airbnb file, and it is large. The listing file already has six aspect scores. I used those and left the raw reviews alone. `review_scores_value` was excluded on purpose: guests score value against the price, so the column leaks the label in the polite direction.

The ad text is on the listing file. I tried a fixed lexicon and a TF-IDF bigram model compressed to eight components, with the vocabulary fit inside the training fold so the test listings could not add words. Neither cleared a 1% MAE bar, alone or on top of the detail block. A sentence-transformer embedding would have been another download and a worse interview explanation of the same text, after the linear compression of that text had failed. I did not fit one.

**Photos.** Multimodal student projects, including Pandey's 2024 thesis on Airbnb pricing, feed listing photos to a vision model and sometimes gain accuracy. The pixels are not in the CC BY dump. The column is a URL to a photograph the host took. I did not download them.

**Spatial lag, geographic weighting, and another smooth of nearby prices.** Almeida and colleagues put a 1 km spatial weight on other listings' prices. Under this study's buffer, those neighbours are exactly the rows the split removes, so a 1 km lag cannot be computed at test time without breaking the split. v2 had already measured a neighbourhood mean and a 40-neighbour median, fit inside the training fold only. The neighbourhood mean was harmful. The neighbour median missed the bar. This round tried one Gaussian kernel, bandwidth 5 km, 80 neighbours, own price removed. It missed the bar too. Three versions of "what do the places over there charge" have failed.

A multiscale geographically weighted regression, as in Zhao, Wu, Chen, and Chen (2023) on Beijing, lets every slope change across the map. It also does not travel to a held-out neighbourhood: the local fit is the neighbourhood. The boosted trees already split on coordinates. I did not fit a GWR.

**One pooled model.** City-by-city fits are what Gibbs et al. report, because the slopes differ. A pooled model with a city indicator was the cheap way to borrow Seattle's small training set from the other two cities. It was about 0.9% better on development MAE. The bar was 1%. It was not kept.

**A wider hyperparameter search.** The tree depth was re-checked on the kept columns. Depth 4 still won, and depth 8 was worse. Learning rate, the number of leaves, and the TF-IDF width were not searched. A search on the same five seeds would have been another chance to overfit the development split. The fixed settings are in `evaluation/protocol.py` and `models/regressors.py`.

**Availability and the calendar.** Inside Airbnb says a night marked unavailable was not necessarily booked. Estimated occupancy and estimated revenue move with the price. None of those columns were eligible.

## Sources

- Almeida, A., Nunes, A. P., and Pinto Machado, L. (2025). How do reviews impact Airbnb's prices? A hedonic approach. *Tourism and Hospitality*, 6(4), 181. https://doi.org/10.3390/tourhosp6040181
- Gibbs, C., Guttentag, D., Gretzel, U., Morton, J., and Goodwill, A. (2018). Pricing in the sharing economy: a hedonic pricing model applied to Airbnb listings. *Journal of Travel & Tourism Marketing*, 35(1), 46–56. https://doi.org/10.1080/10548408.2017.1308292
- Inside Airbnb. Data policies and the CC BY 4.0 listing files. https://insideairbnb.com/data-policies/
- Kalehbasti, P. R., Nikolenko, L., and Rezaei, H. (2019). Airbnb price prediction using machine learning and sentiment analysis. arXiv:1907.12665. Later in *Machine Learning and Knowledge Extraction*, CD-MAKE 2021.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., and Wasserman, L. (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association*, 113(523), 1094–1111.
- Lundberg, S. M., and Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Pandey, A. (2024). Understanding the impact of multimodal models on Airbnb pricing. Master's thesis, National College of Ireland. https://norma.ncirl.ie/9444/
- Papadopoulos, H., Proedrou, K., Vovk, V., and Gammerman, A. (2002). Inductive confidence machines for regression. *European Conference on Machine Learning*.
- Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., and Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40, 913–929. https://doi.org/10.1111/ecog.02881
- Romano, Y., Patterson, E., and Candès, E. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems*, 32. https://arxiv.org/abs/1905.03222
- Rosen, S. (1974). Hedonic prices and implicit markets: product differentiation in pure competition. *Journal of Political Economy*, 82(1), 34–55.
- Wang, D., and Nicolau, J. L. (2017). Price determinants of sharing economy based accommodation rental: a study of listings from 33 cities on Airbnb.com. *International Journal of Hospitality Management*, 62, 120–131.
- Zhao, C., Wu, Y., Chen, Y., and Chen, G. (2023). Multiscale effects of hedonic attributes on Airbnb listing prices based on MGWR: a case study of Beijing, China. *Sustainability*, 15(2), 1703. https://doi.org/10.3390/su15021703
