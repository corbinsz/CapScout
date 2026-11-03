# Production deploy

The owner lives in Utah. Deploy does not depend on a Colorado region. Pick the Render and Vercel regions you already use. Denver in the app is a scored market, not the company location.

Do not put keys in git. Set them in the host’s dashboard.

## 1. Backend on Render

The service is described in `render.yaml`. Root directory `backend`. Start command is already gunicorn.

Set these environment variables:

| Name | Required | Notes |
| --- | --- | --- |
| `SECRET_KEY` | Yes, before real accounts | Long random string. Logins die on restart without it. |
| `RENTCAST_API_KEY` | Yes, for live listings | From the RentCast dashboard. |
| `CORS_ORIGINS` | Yes | The Vercel site origin, for example `https://capscout.vercel.app`. No trailing slash. Add the preview host pattern only if you serve the API directly to the browser. The Vercel rewrite below makes the browser call `/api` on the same site, so CORS is less important for the preview. |
| `SESSION_COOKIE_SECURE` | Yes in production | `1` |
| `GOOGLE_MAPS_API_KEY` | No | Street View. The app works without photos. |
| `STRIPE_SECRET_KEY` | No | `sk_test_...` or `sk_live_...`. Leave empty until you want charges. |
| `STRIPE_PRICE_ID` | With Stripe | `price_...` for the Pro plan. |
| `STRIPE_WEBHOOK_SECRET` | With Stripe | So a paid checkout can mark the account Pro. |
| `ALLOW_REDFIN_FALLBACK` | Leave unset | `1` only on a private dev machine. |
| `ALLOW_REDFIN_DOWNLOAD` | Leave unset | `1` only if you intentionally call Redfin’s unofficial endpoint. |
| `CAPSCOUT_ALLOW_KEY_WRITE` | Leave unset | Lets the browser POST an API key. Do not enable on the public server. |
| `CAPSCOUT_ALLOW_DEV_BILLING` | Leave unset | Lets a signed-in user become Pro with no card. Tests use it. Production must not. |
| `ANALYTICS_ENABLED` | No | `1` to store daily event counts. |
| `CLIENT_ERROR_LOG` | No | `1` to append short error lines under `backend/data/`. |
| `RENTCAST_MAX_PAGES` | No | Default `1` (up to 500 listings, one billed call, cached). Raise it on a paid RentCast plan. |

The account database is created on the server disk. On Render’s free disk it can disappear when the instance is replaced. Move it to a persistent disk before you charge customers. There is no self-serve delete-account button yet. That is called out on the privacy page.

## 2. Frontend on Vercel

Framework preset Vite. Build command `npm run build`. Output `dist`. `vercel.json` already rewrites `/api` to `https://capscout.onrender.com/api` and sends every other path to `index.html`.

A pull-request preview uses that same file, so the new pages load and API calls go to whatever is deployed on Render. Deal behavior changes only after Render is running this branch.

The site works with no API. The Vercel preview reads `public/demo/austin.json` in the browser. Leave the live flags unset there.

Optional build-time variables:

| Name | Effect |
| --- | --- |
| `VITE_USE_LIVE_API` | Set to `1` to call the Python server instead of the Austin sample. |
| `VITE_API_URL` | Required with the flag above. Example: `https://capscout.onrender.com/api`. A cold start is retried once, and the message says the server is waking up. |
| `VITE_ANALYTICS` | `1` to send page and click counts. Also set `ANALYTICS_ENABLED` on the API. |
| `VITE_ERROR_REPORTING` | `1` to send short error messages. Also set `CLIENT_ERROR_LOG` on the API. |

No third-party analytics script is loaded by default.

## 3. Stripe, when you are ready

1. Create a product and a recurring price in Stripe.
2. Set `STRIPE_SECRET_KEY` and `STRIPE_PRICE_ID`.
3. Point a webhook at `https://<your-api>/api/billing/webhook` for `checkout.session.completed`.
4. Set `STRIPE_WEBHOOK_SECRET`.
5. Install the `stripe` Python package on the server if it is not already in the environment (`pip install stripe`). It is imported only when a real key is present.

Until those are set, the upgrade button explains that checkout is not connected and does not collect a card.

## 4. Email alerts

The app stores weekly alert requests. It does not send them. When you pick a provider, read the `alerts` table and send one message per row. Do not log the address. The product copy already tells the person that nothing was sent.

## 5. Check after deploy

Open the checklist in `docs/REVIEW-CHECKLIST.md`. Hit `GET /api/health`. It should return `"status": "ok"` and `"listing_source": "rentcast"` once the key is set.
