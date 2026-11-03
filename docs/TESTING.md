# How to verify CapScout

Run these before you trust a change. They are the same commands as GitHub Actions in `.github/workflows/ci.yml`.

## Backend

From `backend/`, with Python 3.12:

```bash
python -m pip install -r requirements.txt pytest mypy
python -m mypy
python -m pytest -q
```

What that covers:

- Mortgage math and the 0–100 score, including the “not financial advice” sentence.
- City names reject path traversal. Email and password rules.
- RentCast and Redfin rows keep a half bath and drop land.
- The dollar wrapper can be saved and loaded. The Austin model returns a nightly rate in dollars, not a log.
- Holdout groups do not share a listing.
- API: health, unknown city, refusal to save an API key from the browser, register, login, paywall, dev upgrade, CSV download, alert stored but not sent, rate limit, a real Asheville deal list, Stripe stub, analytics allow-list.

The deal-list test turns on `ALLOW_REDFIN_FALLBACK` so it can read the CSV already in the repo. It does not call RentCast or Redfin over the network.

## Frontend

From the repo root, with Node 22:

```bash
npm ci
npm run lint
npm run typecheck
npm run build
```

`typecheck` is TypeScript checking the small modules that talk to the API and the score labels. The rest of the UI is checked by ESLint, including the React hooks rules.

## In the browser

Follow `docs/REVIEW-CHECKLIST.md`. That is the end-to-end pass: home, city pick, deal sentence, share link, cities table, map, saved homes, account, paywall, and a narrow screen.

## Model numbers

`docs/ml_benchmark.json` is the record of the comparison. Re-running it retrains every city and takes a few minutes:

```bash
cd backend
python -m models.benchmark
```

Do not run that on a whim. It replaces the model files.
