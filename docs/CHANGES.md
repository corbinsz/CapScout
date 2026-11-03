# What changed, and why

## Second pass

- `backend/models/occupancy.py` — San Francisco review model. Reviews in the last year divided by 0.5, times the stay (city average, or the listing minimum up to 14 nights), divided by 365, capped at 70%. A full calendar with no reviews is not a booking.
- `backend/models/feature_engineering.py` — training occupancy now uses that model. The old calendar rate is kept as `occupancy_calendar` for the comparison.
- `backend/models/occupancy_benchmark.py` and `docs/occupancy_benchmark.json` — holdout on four cities. The review target is more learnable (error ratio 0.89 vs 0.96). Calendar and review occupancy correlate at about 0.05. Occupancy models for all 14 cities were retrained. Price models were not.
- `docs/OCCUPANCY.md` — the plain-language writeup.
- `docs/UTAH.md` — Salt Lake City and Park City are not on Inside Airbnb’s public list. The path to add them is a funded Inside Airbnb region request, or a licensed AirROI extract. The market was not faked.
- `backend/tests/test_scoring_math.py` — cap rate, cash-on-cash, and DSCR checked against a hand-worked Austin deal, plus a case where debt coverage is below 1.
- `backend/tests/test_occupancy.py` — the review formula, the 14-night cap, the 70% cap, and a blocked calendar.
- Deal cards now say a home can rank at the top of a search and still be short of covering the mortgage. Map tiles are labeled so they are not mistaken for photos. Saved homes has an empty state that points back to deals. Repeated “Unit G2 Unit G2” addresses are cleaned up for display without changing the link id.

## First pass

Grouped by area. Model binaries are listed once: every city under `backend/models/trained/` was retrained and the old monthly feature files were removed.

## Data providers

- `backend/services/providers/registry.py` — RentCast when a key is set. Saved Redfin CSV only if `ALLOW_REDFIN_FALLBACK=1`. No silent hop from a failed licensed call onto Redfin.
- `backend/services/providers/rentcast_provider.py` — licensed client, disk cache, normalizes half baths, drops land.
- `backend/services/providers/redfin_provider.py` — reads a CSV already on disk. Does not call Redfin.
- `backend/services/providers/normalize.py` — shared id and number parsing.
- `backend/services/redfin_service.py` — bathrooms stay floats.
- `backend/scripts/download_redfin.py` — exits unless `ALLOW_REDFIN_DOWNLOAD=1`.
- `backend/config.py` — env flags, cache directory, the missing `DATA_PROCESSED_DIR` name, city paths that reject unknown keys.

## Scoring and API

- `backend/routes/properties.py` — score every filtered home, rank them, add the plain-language line and the “beats N%” line, validate query numbers, rate-limit.
- `backend/services/prediction_service.py` — batch prediction, and derived features if a model asks for them.
- `backend/services/scoring_service.py` — `explain_deal`.
- `backend/routes/predict.py` — bounded inputs and the same sentence.
- `backend/routes/analytics.py` — known cities only, reports which listing source is on.
- `backend/routes/settings.py` — browser cannot save API keys unless a dev flag is on. Redfin upload is off unless the fallback flag is on, and files over 20 MB are refused.
- `backend/routes/markets.py` — highest score in each city, same price band, cached.
- `backend/routes/reports.py` — Pro CSV.
- `backend/routes/model_metrics.py` — holdout numbers saved next to each model.
- `backend/security.py` — city check, number bounds, email and password checks, rate limit.
- `backend/app.py` — cookie settings, limited CORS, security headers, debug off unless `FLASK_DEBUG=1`.

## Accounts

- `backend/services/account_service.py` — hashed passwords, alert rows, daily event counts.
- `backend/routes/auth.py` — register, login, logout. Failed login uses one message. A hidden company field rejects bots.
- `backend/routes/billing.py` — Stripe checkout when keys look real, otherwise a stub. Webhook checks the signature. Dev upgrade is 404 unless the flag is on.
- `backend/routes/alerts.py` — stores a weekly alert and says nothing was sent.
- `backend/routes/telemetry.py` — optional counts and optional error lines. Off by default.

## Models

- `backend/models/metrics.py` — holdout helpers.
- `backend/models/benchmark.py` — compared ridge, elastic net, histogram gradient boosting, XGBoost, XGBoost with extra spatial features, and XGBoost trained on area medians. Kept plain XGBoost because it had the lowest raw nightly-rate error on the holdout and on the time split.
- `backend/models/estimators.py` — saved models return dollars. Loading them no longer recurses.
- `backend/models/trained/<city>/` — new price and occupancy models, feature metadata, and `metrics.json`.
- `docs/ml_benchmark.json` — the before and after numbers.

Austin nightly-rate error on the holdout went from about $117 (old model, and that model had trained on those rows) to about $68 (new model, rows it did not fit). Other cities moved the same direction. Occupancy error barely moved. The full table is in the JSON file.

## Website

- `src/pages/Home.jsx` — first screen. The question, one button, a city grid. No account.
- `src/pages/Dashboard.jsx` — moved to `/deals`. Report button, score legend, holdout sentence when metrics exist.
- `src/pages/Markets.jsx` — cross-city table.
- `src/pages/PropertyPage.jsx` — shared `?city=` links, share button, insight line. Fetch no longer fights the linter by setting loading before the request returns.
- `src/pages/Settings.jsx` — explains RentCast and the Redfin flag. No key boxes.
- `src/pages/Terms.jsx`, `src/pages/Privacy.jsx` — placeholders, marked as such.
- `src/pages/PredictionProcess.jsx` — the old R-squared table is labeled do-not-quote, and each score piece has a one-line meaning.
- `src/components/layout/Navbar.jsx` — Deals, Cities, Map, Saved, Sign in.
- `src/components/layout/Layout.jsx`, `Footer.jsx` — skip link, disclaimer, terms.
- `src/components/auth/AuthModal.jsx`, `PaywallModal.jsx` — account and the Pro stub.
- `src/components/alerts/AlertForm.jsx` — weekly alert request.
- `src/components/properties/PropertyCard.jsx` — score word, sentence, percentile line, city on the link.
- `src/components/filters/FilterPanel.jsx` — filter buttons expose whether they are open.
- `src/contexts/AuthContext.jsx` — session user.
- `src/analytics.js`, `src/monitoring.js`, `src/components/common/ErrorBoundary.jsx` — off unless the build flags are set.
- `src/utils/markets.js`, `src/utils/scoreHelpers.js` — city list and score words (Fair, Weak).
- `src/index.css` — landing, table, focus ring, modals, footer.
- `src/App.jsx`, `src/main.jsx`, `src/services/api.js` — routes, error boundary, cookies.
- `index.html`, `public/robots.txt`, `public/sitemap.xml` — title, description, canonical, sitemap.
- `vercel.json` — preview and production send `/api` to the Render service, then fall through to the app.
- `eslint.config.js`, `tsconfig.json`, `src/env.d.ts`, `package.json` — lint and a small typecheck.
- A few existing screens were adjusted only so lint could pass: `Compare.jsx`, `useAnalytics.js`, `useFilters.js`, `FavoritesContext.jsx`, `RevenueChart.jsx`, `TrainingData.jsx`.

## Tests and docs

- `backend/tests/` — unit and API tests.
- `backend/mypy.ini`, `backend/pytest.ini` — type check scope and pytest.
- `.github/workflows/ci.yml` — lint, typecheck, build, mypy, pytest.
- `docs/ARCHITECTURE.md`, `PRODUCT.md`, `TESTING.md`, `DEPLOY.md`, `AUDIT.md`, `REVIEW-CHECKLIST.md`, this file.
- `README.md` — for-sale source and the accuracy section no longer claim a 0.97 holdout.

## Not done, on purpose

- No live Stripe charge and no live email. Both need keys the owner chooses.
- Salt Lake City was not added. There is no training file for it.
- The map bundle is still large. It loads only on the map and property pages.
- Account deletion is described on the privacy page and is not a button yet.
