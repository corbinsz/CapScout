import { useEffect, useState } from 'react';
import { useCity } from '../contexts/CityContext';
import { predictProperty } from '../services/propertyService';
import { formatCurrency } from '../utils/formatters';
import api from '../services/api';

function PredictionTryout({ cityKey, latitude, longitude }) {
  const [testForm, setTestForm] = useState({
    purchase_price: 450000,
    bedrooms: 3,
    bathrooms: 2,
    latitude,
    longitude,
  });
  const [testResult, setTestResult] = useState(null);
  const [testError, setTestError] = useState(null);
  const [testing, setTesting] = useState(false);

  const handleTest = async () => {
    setTesting(true);
    setTestError(null);
    try {
      const result = await predictProperty({ ...testForm, city: cityKey });
      setTestResult(result);
    } catch (err) {
      setTestError(err.response?.data?.error || err.message);
    }
    setTesting(false);
  };

  return (
    <div className="settings-form">
      <div className="form-group">
        <label htmlFor="try-price">Purchase Price ($)</label>
        <input id="try-price" type="text" value={testForm.purchase_price} onChange={e => setTestForm(f => ({ ...f, purchase_price: Number(e.target.value) }))} />
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <div className="form-group">
          <label htmlFor="try-beds">Bedrooms</label>
          <input id="try-beds" type="text" value={testForm.bedrooms} onChange={e => setTestForm(f => ({ ...f, bedrooms: Number(e.target.value) }))} />
        </div>
        <div className="form-group">
          <label htmlFor="try-baths">Bathrooms</label>
          <input id="try-baths" type="text" value={testForm.bathrooms} onChange={e => setTestForm(f => ({ ...f, bathrooms: Number(e.target.value) }))} />
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <div className="form-group">
          <label htmlFor="try-lat">Latitude</label>
          <input id="try-lat" type="text" value={testForm.latitude} onChange={e => setTestForm(f => ({ ...f, latitude: Number(e.target.value) }))} />
        </div>
        <div className="form-group">
          <label htmlFor="try-lng">Longitude</label>
          <input id="try-lng" type="text" value={testForm.longitude} onChange={e => setTestForm(f => ({ ...f, longitude: Number(e.target.value) }))} />
        </div>
      </div>
      <button className="btn btn-primary" type="button" onClick={handleTest} disabled={testing}>
        {testing ? 'Predicting...' : 'Run Prediction'}
      </button>
      {testError && <p className="form-error">{testError}</p>}
      {testResult && (
        <div style={{ marginTop: 16 }}>
          <p className="why-line">{testResult.explanation}</p>
          <div className="detail-stats">
            <div className="detail-stat">
              <span className="stat-label">CapScout Score</span>
              <span className="stat-value" style={{ color: 'var(--primary)' }}>{testResult.score?.toFixed(1)}</span>
            </div>
            <div className="detail-stat">
              <span className="stat-label">Pred. Nightly</span>
              <span className="stat-value">{formatCurrency(testResult.prediction?.predicted_nightly_rate)}</span>
            </div>
            <div className="detail-stat">
              <span className="stat-label">Pred. Monthly</span>
              <span className="stat-value">{formatCurrency(testResult.prediction?.predicted_monthly_revenue)}</span>
            </div>
            <div className="detail-stat">
              <span className="stat-label">Cap Rate</span>
              <span className="stat-value">{testResult.financials?.cap_rate}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function Settings() {
  const { currentCity, cityMeta } = useCity();
  const [keyStatus, setKeyStatus] = useState({});

  useEffect(() => {
    let cancelled = false;
    api.get('/settings/keys', { params: { city: currentCity } })
      .then(res => { if (!cancelled) setKeyStatus(res.data); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [currentCity]);

  const cityDisplay = cityMeta?.display || 'Austin, TX';
  const sourceLabel = keyStatus.listing_label || 'Not configured';

  return (
    <div>
      <div className="page-header">
        <h1>Data sources</h1>
        <p>Where {cityDisplay} listings come from, and a way to try the model on numbers you type in.</p>
      </div>

      <div className="detail-section" style={{ marginBottom: 24 }}>
        <div className="section-header">Listing data</div>
        <div className="section-body legal-copy">
          <p>
            Active source: <strong>{sourceLabel}</strong>
            {keyStatus.listing_licensed ? ' (licensed)' : keyStatus.listing_source === 'demo' ? ' (saved sample)' : keyStatus.listing_source ? ' (development fallback)' : ''}
          </p>
          <p>
            For-sale homes are meant to come from RentCast. Put <code>RENTCAST_API_KEY</code> in the
            server environment. The key is never typed into this page and is never sent to the browser.
          </p>
          <p>
            A saved Redfin file can be read only when the server has <code>ALLOW_REDFIN_FALLBACK=1</code>.
            That path is for local development. Redfin&apos;s unofficial map feed is not licensed for a
            commercial product, and this app will not call it unless <code>ALLOW_REDFIN_DOWNLOAD=1</code> as well.
          </p>
          <p>
            Short-term rental history used to train the models is from Inside Airbnb, under the
            Creative Commons Attribution 4.0 license. Credit Inside Airbnb when you publish numbers from it.
            Photos and listing text stay on Airbnb.
          </p>
          {keyStatus.redfin_fallback_enabled && (
            <p>{keyStatus.redfin_count || 0} rows in the saved development file for {cityDisplay}.</p>
          )}
        </div>
      </div>

      <div className="detail-section">
        <div className="section-header">Try a prediction</div>
        <div className="section-body">
          <PredictionTryout
            key={currentCity}
            cityKey={currentCity}
            latitude={cityMeta?.latitude || 30.27}
            longitude={cityMeta?.longitude || -97.74}
          />
        </div>
      </div>
    </div>
  );
}
