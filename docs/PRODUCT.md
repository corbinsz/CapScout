# Why someone would pay for CapScout

CapScout is a screener. An investor picks a city and sees which homes for sale could carry a mortgage as a short-term rental, with the reason written in a sentence. That is the thing a person cannot do quickly with a listing site plus a spreadsheet.

The owner lives in Utah. The product does not assume a Colorado home base. Denver is one of 14 scored markets. Salt Lake City is not scored yet, because this repo has no trained model or Inside Airbnb snapshot for it.

## What is hard to copy

**The score is not a nightly rate.** Five pieces (gross yield, cash-on-cash, cap rate, debt coverage, occupancy) use the same weights and the same loan math in every city. A 70 in Austin means the same kind of deal as a 70 in Nashville. Copying the screen is easy. Keeping the expense model, city taxes, insurance, permits, and cleaning costs consistent is the work.

**Cross-city ranking uses that one score.** The Cities page asks each market for its best home between $150,000 and $750,000 and sorts those homes on the same scale. A listing site will not do that, because the sites do not share a short-term-rental underwriting model.

**The error is published.** Each city has a holdout: whole listings the model did not train on, plus a split by review date. The old “0.97 R-squared” figure was an in-sample fit on area-median prices. It is labeled as history in the app and in the README. A copycat who quotes a pretty R-squared is easier to beat than a product that shows a dollar miss.

**Data has a license path.** For-sale homes are meant to come from RentCast. Redfin’s unofficial map feed is refused unless a developer turns on an explicit local flag. Inside Airbnb’s downloadable files are under Creative Commons Attribution 4.0. Commercial use is allowed with credit. Inside Airbnb also asks people not to republish the raw files and not to scrape the site. Photos and guest text stay out of CapScout. Details are in ARCHITECTURE.md.

**Alerts are a hook, not a mailbox.** A person can ask for a weekly note when a home in a city scores 70 or higher. The app stores that request and does not send mail until an email provider is connected. The hard part later is delivering the right homes without emailing the whole market.

## What a free visitor gets

They pick a city and see scores, cash flow, and a sentence. No account.

## What Pro is for

Pro downloads a CSV of the top deals in the city on screen. Checkout is wired for Stripe and does nothing until `STRIPE_SECRET_KEY` and `STRIPE_PRICE_ID` are set. No card is charged in this repository.

## What is not a moat

The code is in this repo. A determined person can read the score weights. The advantage is the maintained data, the honest error, the same math across cities, and the habit of opening CapScout instead of rebuilding the spreadsheet.
