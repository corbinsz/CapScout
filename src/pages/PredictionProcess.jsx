export default function PredictionProcess() {
  return (
    <div>
      <div className="page-header">
        <h1>How the model works</h1>
        <p>
          CapScout is a personal project, built in spare time for fun and learning. It is not a commercial product.
          The numbers that matter are in <code>reports/FINDINGS.md</code> in the repo.
        </p>
      </div>

      <div className="detail-section" style={{ marginBottom: 24 }}>
        <div className="section-header">
          What is being predicted
          <div className="section-subtitle">A listing's own advertised nightly rate</div>
        </div>
        <div className="section-body" style={{ fontSize: 14, lineHeight: 1.8 }}>
          <p>
            The target is the price on the Inside Airbnb listing, in dollars, after training on the log of that
            price. It is not a neighbourhood median. An older version of this project trained on area medians
            that were built from the same prices, and the R² looked excellent because the target had already
            been smoothed. That number is not used.
          </p>
          <p>
            The study markets are Austin, San Diego, and Seattle. Other city files are still in the data
            folder and are not part of the headline results. The write-up explains why these three.
          </p>
        </div>
      </div>

      <div className="detail-section" style={{ marginBottom: 24 }}>
        <div className="section-header">
          What the local screen is showing
          <div className="section-subtitle">One rate, plus an illustrative calendar</div>
        </div>
        <div className="section-body" style={{ fontSize: 14, lineHeight: 1.8 }}>
          <p>
            The saved model predicts a single nightly rate from size, room type, location, and — if the
            study kept them — listing details such as amenities and reviews. For a home that is not yet
            listed, reviews and amenities are left blank rather than filled in as zeros.
          </p>
          <p>
            The 12-month curve multiplies that rate by hand-set seasonal weights that average to 1.
            They are a sketch of the travel calendar, not something the model learned.
            Occupancy on this screen is the neighbourhood median of a review-based proxy
            (Inside Airbnb's San Francisco model, capped at 70%). It is not observed bookings,
            and it is not a second machine-learning model.
          </p>
          <p>
            The CapScout score then runs that revenue through a simple mortgage worksheet
            (20% down, a fixed rate, city tax and insurance assumptions). It is a calculator
            on top of the prediction, not a measure of model accuracy.
          </p>
        </div>
      </div>
    </div>
  );
}
