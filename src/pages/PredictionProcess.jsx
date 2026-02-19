export default function PredictionProcess() {
  return (
    <div>
      <div className="page-header">
        <h1>Prediction Process</h1>
        <p>How CapScout predicts short-term rental performance for any property</p>
      </div>

      {/* Step 1: Data Collection */}
      <div className="detail-section" style={{ marginBottom: 24 }}>
        <div className="section-header">
          Step 1: Data Collection
          <div className="section-subtitle">Where the training data comes from</div>
        </div>
        <div className="section-body" style={{ fontSize: 14, lineHeight: 1.8 }}>
          <p style={{ marginBottom: 12 }}>
            CapScout trains its models on real Airbnb performance data from{' '}
            <strong>Inside Airbnb</strong>, a public dataset that provides every active listing's
            nightly rate, occupancy, property type, location, and calendar availability.
          </p>
          <p style={{ marginBottom: 12 }}>
            For cities with available county assessor records (currently Austin, Seattle, and Dallas),
            the training data is <strong>enriched with physical property attributes</strong> &mdash;
            square footage, year built, and lot size &mdash; by spatially matching Airbnb listings
            to tax assessor parcels using nearest-neighbor search within ~200 meters.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
            <div style={{
              background: 'var(--primary-light)', borderRadius: 'var(--radius-md)',
              padding: '16px', borderLeft: '4px solid var(--primary)'
            }}>
              <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 6, color: 'var(--primary)' }}>
                Airbnb Listings + Calendar
              </div>
              <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                Nightly rates, occupancy patterns, property type, bedrooms, bathrooms,
                location, and monthly booking data for thousands of active listings per city.
              </div>
            </div>
            <div style={{
              background: 'var(--teal-light)', borderRadius: 'var(--radius-md)',
              padding: '16px', borderLeft: '4px solid var(--teal)'
            }}>
              <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 6, color: 'var(--teal)' }}>
                County Assessor Records
              </div>
              <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                Square footage, year built, lot size, and bedroom count from public tax records.
                Matched to Airbnb listings by proximity and bedroom count.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Step 2: Data Cleaning */}
      <div className="detail-section" style={{ marginBottom: 24 }}>
        <div className="section-header">
          Step 2: Data Cleaning
          <div className="section-subtitle">Filtering noise from the training data</div>
        </div>
        <div className="section-body" style={{ fontSize: 14, lineHeight: 1.8 }}>
          <p style={{ marginBottom: 12 }}>
            Raw Airbnb data is noisy. The cleaning pipeline applies several filters to get a
            reliable training signal:
          </p>
          <ul style={{ paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <li><strong>Entire homes only</strong> &mdash; private rooms and shared spaces are excluded since for-sale properties are whole units</li>
            <li><strong>Listing price cap (P2-P98)</strong> &mdash; removes extreme outlier prices from the listing data</li>
            <li><strong>Calendar price cap (P95)</strong> &mdash; some hosts set $9,999/night as a placeholder to block dates. Capping at the 95th percentile removes these fake prices while keeping legitimately expensive listings</li>
            <li><strong>Minimum data requirements</strong> &mdash; listings must have valid location, at least 1 bedroom, and a calculable occupancy rate</li>
          </ul>
        </div>
      </div>

      {/* Step 3: Area Median Targeting */}
      <div className="detail-section" style={{ marginBottom: 24 }}>
        <div className="section-header">
          Step 3: Area Median Targeting
          <div className="section-subtitle">The key insight that makes predictions accurate</div>
        </div>
        <div className="section-body" style={{ fontSize: 14, lineHeight: 1.8 }}>
          <p style={{ marginBottom: 12 }}>
            Two identical 3-bedroom homes on the same street can list at $150/night and $400/night
            depending on the host's pricing strategy, photo quality, and marketing. Training a model
            to predict individual host pricing from property features alone is unreliable.
          </p>
          <p style={{ marginBottom: 16 }}>
            CapScout solves this with <strong>area median targeting</strong>: instead of predicting
            what each individual host charges, the model predicts what a <strong>typical listing
            in that area earns</strong>.
          </p>
          <div style={{
            background: 'var(--bg)', borderRadius: 'var(--radius-md)', padding: 20,
            border: '1px solid var(--border)'
          }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                <span style={{
                  background: 'var(--primary)', color: '#fff', borderRadius: '50%',
                  width: 28, height: 28, display: 'flex', alignItems: 'center',
                  justifyContent: 'center', fontWeight: 700, fontSize: 13, flexShrink: 0
                }}>1</span>
                <div>
                  <strong>Cluster</strong> all listings into ~20 geographic zones using k-means on coordinates
                </div>
              </div>
              <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                <span style={{
                  background: 'var(--teal)', color: '#fff', borderRadius: '50%',
                  width: 28, height: 28, display: 'flex', alignItems: 'center',
                  justifyContent: 'center', fontWeight: 700, fontSize: 13, flexShrink: 0
                }}>2</span>
                <div>
                  <strong>Group</strong> listings by (zone, bedroom count, month) &mdash; e.g., "3-bedrooms in East Austin in July"
                </div>
              </div>
              <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                <span style={{
                  background: 'var(--indigo)', color: '#fff', borderRadius: '50%',
                  width: 28, height: 28, display: 'flex', alignItems: 'center',
                  justifyContent: 'center', fontWeight: 700, fontSize: 13, flexShrink: 0
                }}>3</span>
                <div>
                  <strong>Replace</strong> each listing's individual price and occupancy with the group's <strong>median</strong> as the training target
                </div>
              </div>
            </div>
          </div>
          <p style={{ marginTop: 16, color: 'var(--text-muted)', fontSize: 13 }}>
            This eliminates host-specific noise. For occupancy, it also mitigates the problem of
            hosts blocking calendar dates (which looks identical to zero bookings). The model
            learns: "what can you expect a typical STR at this location to earn?"
          </p>
        </div>
      </div>

      {/* Step 4: Feature Selection */}
      <div className="detail-section" style={{ marginBottom: 24 }}>
        <div className="section-header">
          Step 4: Feature Selection (17 Features + 2 Seasonal)
          <div className="section-subtitle">Only features available for both Airbnb training and Redfin for-sale properties</div>
        </div>
        <div className="section-body" style={{ fontSize: 14, lineHeight: 1.8 }}>
          <p style={{ marginBottom: 16 }}>
            Since predictions are made on Redfin for-sale properties, only features that exist for
            <strong> both</strong> Airbnb listings and Redfin listings are included. Amenities, review
            scores, host metrics, and booking activity were all removed &mdash; they would be filled
            with constants for for-sale properties, adding noise without signal.
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table className="comps-table" style={{ fontSize: 13 }}>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Source (Redfin)</th>
                  <th>Type</th>
                </tr>
              </thead>
              <tbody>
                <tr><td>bedrooms</td><td>BEDS</td><td>real</td></tr>
                <tr><td>bathrooms</td><td>BATHS</td><td>real</td></tr>
                <tr><td>accommodates</td><td>bedrooms x 2</td><td>derived</td></tr>
                <tr><td>latitude, longitude</td><td>LATITUDE, LONGITUDE</td><td>real</td></tr>
                <tr><td>distance_to_center</td><td>haversine from city center</td><td>derived</td></tr>
                <tr><td>is_home, is_condo, is_townhouse, is_cabin, is_cottage</td><td>PROPERTY TYPE</td><td>flags</td></tr>
                <tr><td>sqft</td><td>SQUARE FEET</td><td>real</td></tr>
                <tr><td>year_built</td><td>YEAR BUILT</td><td>real</td></tr>
                <tr><td>lot_sqft</td><td>LOT SIZE</td><td>real</td></tr>
                <tr><td>sqft_per_bedroom</td><td>sqft / bedrooms</td><td>derived</td></tr>
                <tr><td>property_age</td><td>2026 - year_built</td><td>derived</td></tr>
                <tr><td>lot_acres</td><td>lot_sqft / 43560</td><td>derived</td></tr>
                <tr><td>sin_month, cos_month</td><td>month (1-12)</td><td>seasonal</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Step 5: Model Training */}
      <div className="detail-section" style={{ marginBottom: 24 }}>
        <div className="section-header">
          Step 5: XGBoost Model Training
          <div className="section-subtitle">Two models per city: price and occupancy</div>
        </div>
        <div className="section-body" style={{ fontSize: 14, lineHeight: 1.8 }}>
          <p style={{ marginBottom: 12 }}>
            CapScout trains <strong>two XGBoost models per city</strong>:
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
            <div style={{
              background: 'var(--primary-light)', borderRadius: 'var(--radius-md)',
              padding: '16px', borderLeft: '4px solid var(--primary)'
            }}>
              <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4, color: 'var(--primary)' }}>
                Price Model
              </div>
              <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                Predicts expected nightly rate from property features, location, and month
              </div>
            </div>
            <div style={{
              background: 'var(--teal-light)', borderRadius: 'var(--radius-md)',
              padding: '16px', borderLeft: '4px solid var(--teal)'
            }}>
              <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4, color: 'var(--teal)' }}>
                Occupancy Model
              </div>
              <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                Predicts monthly occupancy rate with seasonal awareness (captures booking patterns)
              </div>
            </div>
          </div>
          <p style={{ marginBottom: 12 }}>
            Hyperparameters are tuned via <strong>RandomizedSearchCV</strong> (50 random
            combinations, 5-fold cross-validation). The best parameters are used to retrain
            on <strong>100% of the data</strong> &mdash; there is no hold-out split. The real
            "test set" is Redfin for-sale properties, which come from a completely different
            distribution than the training data.
          </p>
          <p>
            Training reports both <strong>Train R2</strong> (how well the model fits) and{' '}
            <strong>CV R2</strong> (how well it generalizes to unseen data). A small gap
            between them means the model generalizes well.
          </p>

          <div style={{ overflowX: 'auto', marginTop: 16 }}>
            <table className="comps-table" style={{ fontSize: 13 }}>
              <thead>
                <tr>
                  <th>City</th>
                  <th>Price CV R2</th>
                  <th>Occupancy CV R2</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                <tr><td>Honolulu</td><td>0.993</td><td>0.996</td><td></td></tr>
                <tr><td>San Diego</td><td>0.996</td><td>0.971</td><td></td></tr>
                <tr><td>Fort Lauderdale</td><td>0.981</td><td>0.943</td><td></td></tr>
                <tr><td>Las Vegas</td><td>0.980</td><td>0.980</td><td></td></tr>
                <tr><td>Dallas</td><td>0.973</td><td>0.853</td><td>Assessor enriched</td></tr>
                <tr><td>New Orleans</td><td>0.972</td><td>0.947</td><td></td></tr>
                <tr><td>Austin</td><td>0.971</td><td>0.936</td><td>Assessor enriched</td></tr>
                <tr><td>Asheville</td><td>0.965</td><td>0.967</td><td></td></tr>
                <tr><td>Denver</td><td>0.961</td><td>0.925</td><td></td></tr>
                <tr><td>Washington, D.C.</td><td>0.956</td><td>0.925</td><td></td></tr>
                <tr><td>Seattle</td><td>0.951</td><td>0.920</td><td>Assessor enriched</td></tr>
                <tr><td>Boston</td><td>0.932</td><td>0.907</td><td></td></tr>
                <tr><td>Nashville</td><td>0.921</td><td>0.933</td><td></td></tr>
                <tr><td>Fort Worth</td><td>0.886</td><td>0.809</td><td></td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Step 6: For-Sale Prediction */}
      <div className="detail-section" style={{ marginBottom: 24 }}>
        <div className="section-header">
          Step 6: For-Sale Property Prediction
          <div className="section-subtitle">Applying the model to Redfin listings</div>
        </div>
        <div className="section-body" style={{ fontSize: 14, lineHeight: 1.8 }}>
          <p style={{ marginBottom: 12 }}>
            When you view a for-sale property on the dashboard, CapScout:
          </p>
          <ol style={{ paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <li>
              <strong>Extracts features</strong> from the Redfin listing &mdash; bedrooms, bathrooms,
              sqft, year built, lot size, property type, and GPS coordinates
            </li>
            <li>
              <strong>Predicts nightly rate and occupancy for each of the 12 months</strong> &mdash;
              seasonal variation is captured via sin/cos encoding of the month
            </li>
            <li>
              <strong>Computes annual revenue</strong> from the 12 monthly predictions
              (nightly rate x occupancy x days per month, summed across all months)
            </li>
            <li>
              <strong>Runs the financial model</strong> to produce the CapScout Score
            </li>
          </ol>

          <div style={{
            marginTop: 16, padding: 16,
            background: 'var(--indigo-light)', borderRadius: 'var(--radius-md)',
            borderLeft: '4px solid var(--indigo)'
          }}>
            <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 6, color: 'var(--indigo)' }}>
              Prediction Confidence
            </div>
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              Confidence is based on how many Airbnb listings are near the property, weighted by distance
              (1km counts 3x, 3km counts 2x, 5km counts 1x).
              <strong> High</strong> means dense nearby training data,
              <strong> Medium</strong> means moderate coverage,
              <strong> Low</strong> means sparse data in the area.
              By default, the dashboard hides low-confidence properties.
            </div>
          </div>
        </div>
      </div>

      {/* Step 7: Redfin Data Pipeline */}
      <div className="detail-section" style={{ marginBottom: 24 }}>
        <div className="section-header">
          Step 7: Redfin CSV Pipeline
          <div className="section-subtitle">How for-sale listings are imported and filtered</div>
        </div>
        <div className="section-body" style={{ fontSize: 14, lineHeight: 1.8 }}>
          <p style={{ marginBottom: 12 }}>
            For-sale listings are fetched from Redfin's API for each city. The data goes through
            two filter stages:
          </p>
          <div style={{
            background: 'var(--bg)', borderRadius: 'var(--radius-md)', padding: 20,
            border: '1px solid var(--border)'
          }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                <span style={{
                  background: 'var(--primary)', color: '#fff', borderRadius: 'var(--radius-sm)',
                  padding: '2px 10px', fontWeight: 700, fontSize: 12, flexShrink: 0
                }}>RAW</span>
                <div>All rows in the CSV (e.g. 4,290 for Austin)</div>
              </div>
              <div style={{ borderLeft: '2px dashed var(--border)', marginLeft: 26, paddingLeft: 20, paddingTop: 4, paddingBottom: 4, fontSize: 13, color: 'var(--text-muted)' }}>
                Removes: vacant land, commercial, 0-bedroom, no address/price, construction plans
              </div>
              <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                <span style={{
                  background: 'var(--teal)', color: '#fff', borderRadius: 'var(--radius-sm)',
                  padding: '2px 10px', fontWeight: 700, fontSize: 12, flexShrink: 0
                }}>SETTINGS</span>
                <div>Import count shown on Settings page (e.g. 3,735)</div>
              </div>
              <div style={{ borderLeft: '2px dashed var(--border)', marginLeft: 26, paddingLeft: 20, paddingTop: 4, paddingBottom: 4, fontSize: 13, color: 'var(--text-muted)' }}>
                Removes: $0 price (pending, withdrawn, or "coming soon" listings)
              </div>
              <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                <span style={{
                  background: 'var(--indigo)', color: '#fff', borderRadius: 'var(--radius-sm)',
                  padding: '2px 10px', fontWeight: 700, fontSize: 12, flexShrink: 0
                }}>DASHBOARD</span>
                <div>Properties shown and scored (e.g. 2,572)</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Step 8: Scoring */}
      <div className="detail-section" style={{ marginBottom: 24 }}>
        <div className="section-header">
          Step 8: The CapScout Score (0-100)
          <div className="section-subtitle">Weighted composite of five financial metrics</div>
        </div>
        <div className="section-body" style={{ fontSize: 14, lineHeight: 1.8 }}>
          <p style={{ marginBottom: 16 }}>
            The predicted revenue feeds into a financial model that calculates five investment
            metrics. Each is scored on a curve and weighted to produce the final CapScout Score:
          </p>

          <div className="score-breakdown-bars">
            {[
              { label: 'Gross Yield', max: 25, color: 'var(--primary)', desc: 'Annual revenue / purchase price' },
              { label: 'Cash-on-Cash', max: 25, color: 'var(--primary)', desc: 'Annual cash flow / down payment' },
              { label: 'Cap Rate', max: 20, color: 'var(--teal)', desc: 'Net operating income / purchase price' },
              { label: 'DSCR', max: 20, color: 'var(--teal)', desc: 'Debt service coverage (NOI / debt payments)' },
              { label: 'Occupancy', max: 10, color: 'var(--indigo)', desc: 'Predicted booking rate' },
            ].map(({ label, max, color, desc }) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span className="sb-label">{label}</span>
                <div className="sb-track" style={{ flex: 1 }}>
                  <div className="sb-fill" style={{ width: `${max}%`, background: color }} />
                </div>
                <span className="sb-value">{max} pts</span>
              </div>
            ))}
          </div>

          <div style={{
            marginTop: 20, padding: 16,
            background: 'var(--bg)', borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border)', fontSize: 13
          }}>
            <div style={{ fontWeight: 700, marginBottom: 8 }}>Score interpretation</div>
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
              <span><strong style={{ color: 'var(--green)' }}>75+</strong> Excellent</span>
              <span><strong style={{ color: 'var(--teal)' }}>55-74</strong> Good</span>
              <span><strong style={{ color: 'var(--yellow)' }}>40-54</strong> Fair</span>
              <span><strong style={{ color: 'var(--red)' }}>&lt;40</strong> Poor</span>
            </div>
            <div style={{ marginTop: 8, color: 'var(--text-muted)' }}>
              Financial modeling assumes 20% down payment, 7% mortgage rate, 30-year term,
              and city-specific property tax / insurance / expense rates.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
