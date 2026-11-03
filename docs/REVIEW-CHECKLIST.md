# Review checklist

Use this on a Vercel preview of the pull request, or on a local run. The preview site serves the new pages. Deal data comes from the Render API only after that API is deployed from this branch. Until then, the home page, terms, and privacy still load, and deal calls may hit the previous API.

Local run, from the repo root:

```bash
cd backend
ALLOW_REDFIN_FALLBACK=1 SECRET_KEY=local-dev python app.py
```

In another terminal:

```bash
npm install
npm run dev
```

Open http://localhost:3000. The dev server proxies `/api` to port 5000.

The owner lives in Utah. Nothing in the product should assume the company is in Colorado. Denver is only a market you can score.

## Click through

1. Home. In about five seconds you should see the question “Can the rent pay for the house?” and a button for top deals. No login wall.
2. Pick a city that is not the one already highlighted. You should land on Deals for that city with no other setup.
3. On a deal card, read the sentence under the price. It should say what the score means in cash-flow language, and the better homes should say they beat most of the search.
4. Open a home. Confirm the score, the sentence, the breakdown (yield, cash-on-cash, cap rate, debt coverage, occupancy), and the line that this is not financial advice.
5. Share this deal. It should copy or share a link that includes `?city=`. Open that link in a private window and confirm the same home loads.
6. Cities. Wait for the table. Each row is the top score in that city between $150k and $750k. Open one address.
7. Map. Markers should match the city you picked. This page is heavy. Give it a moment.
8. Saved. Star a home, open Saved, confirm it is still there after refresh.
9. Footer. Open How scoring works, Terms, and Privacy. The scoring page should say the old 0.97 R-squared table is not the number to quote.
10. Data sources (footer). It should say RentCast is the licensed feed and that keys are not typed into the browser.
11. Weekly alert on the home page. Submit an email. The reply should say nothing was sent.
12. Create a free account from Sign in. Download “Top deals” before upgrading. You should be asked to upgrade, not given the file.
13. On a machine with `CAPSCOUT_ALLOW_DEV_BILLING=1`, the upgrade dialog has a developer button that marks the account Pro without a card. After that, the CSV downloads and the first lines say it is not financial advice.
14. Narrow the browser to a phone width. The menu opens, the city grid stacks, and the deal cards are one column. Text should not overflow the screen.
15. Tab from the top of the home page. “Skip to content” should appear, and buttons should show a visible focus ring.

## How to request changes

Comment on the pull request with the screen, what you expected, and what you saw. A screenshot is enough. Say if the note is a bug, a wording change, or a product decision (price, which cities, when to charge).

Do not paste API keys, Stripe secrets, or passwords into the pull request or into chat. Put keys only in the host’s environment settings.
