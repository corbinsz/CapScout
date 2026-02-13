import { useState, useEffect } from 'react';
import { predictProperty } from '../services/propertyService';
import { formatCurrency } from '../utils/formatters';
import api from '../services/api';

export default function Settings() {
  const [rentcastKey, setRentcastKey] = useState('');
  const [rapidapiKey, setRapidapiKey] = useState('');
  const [keyStatus, setKeyStatus] = useState({ rentcast: false, rapidapi: false });
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState('');

  // Redfin CSV upload
  const [redfinFile, setRedfinFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState('');

  // Quick test predictor
  const [testForm, setTestForm] = useState({
    purchase_price: 450000,
    bedrooms: 3,
    bathrooms: 2,
    latitude: 30.27,
    longitude: -97.74,
  });
  const [testResult, setTestResult] = useState(null);
  const [testError, setTestError] = useState(null);
  const [testing, setTesting] = useState(false);

  useEffect(() => {
    api.get('/settings/keys').then(res => setKeyStatus(res.data)).catch(() => {});
  }, []);

  const handleSaveKeys = async () => {
    setSaving(true);
    setSaveMsg('');
    try {
      const body = {};
      if (rentcastKey) body.rentcast_key = rentcastKey;
      if (rapidapiKey) body.rapidapi_key = rapidapiKey;
      const res = await api.post('/settings/keys', body);
      setKeyStatus({ rentcast: res.data.rentcast, rapidapi: res.data.rapidapi });
      setSaveMsg('Keys saved successfully.');
      setRentcastKey('');
      setRapidapiKey('');
    } catch (err) {
      setSaveMsg('Failed to save keys: ' + (err.response?.data?.error || err.message));
    }
    setSaving(false);
  };

  const handleRedfinUpload = async () => {
    if (!redfinFile) return;
    setUploading(true);
    setUploadMsg('');
    try {
      const formData = new FormData();
      formData.append('file', redfinFile);
      const res = await api.post('/settings/redfin-upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setKeyStatus(prev => ({ ...prev, redfin: true, redfin_count: res.data.count }));
      setUploadMsg(`Success! ${res.data.count} listings loaded.`);
      setRedfinFile(null);
    } catch (err) {
      setUploadMsg('Upload failed: ' + (err.response?.data?.error || err.message));
    }
    setUploading(false);
  };

  const handleTest = async () => {
    setTesting(true);
    setTestError(null);
    try {
      const result = await predictProperty(testForm);
      setTestResult(result);
    } catch (err) {
      setTestError(err.response?.data?.error || err.message);
    }
    setTesting(false);
  };

  return (
    <div>
      <div className="page-header">
        <h1>Settings</h1>
        <p>Configure API keys and test the prediction model</p>
      </div>

      {/* Redfin CSV Import */}
      <div className="detail-section" style={{ marginBottom: 24 }}>
        <div className="section-header">Redfin CSV Import (Free — No API Key)</div>
        <div className="section-body">
          <div className="settings-form">
            <div style={{ marginBottom: 16, fontSize: 14, lineHeight: 1.7 }}>
              <p style={{ marginBottom: 8, fontWeight: 600 }}>How to get free for-sale listings (no account needed):</p>
              <ol style={{ margin: 0, paddingLeft: 20 }}>
                <li>Go to <strong>redfin.com</strong> and search for a city (e.g. "Austin, TX")</li>
                <li>Apply any filters you want (price, beds, home type, etc.)</li>
                <li>Click the <strong>"Download All"</strong> link near the top-right of the results — it saves a <code>.csv</code> file</li>
                <li>Choose that file below and click <strong>Upload CSV</strong></li>
              </ol>
              <p className="help-text" style={{ marginTop: 8 }}>
                Redfin data replaces any previous upload. Re-upload anytime for fresh listings.
              </p>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <input
                type="file"
                accept=".csv"
                onChange={e => setRedfinFile(e.target.files[0] || null)}
                style={{ flex: 1 }}
              />
              <button
                className="btn btn-primary"
                onClick={handleRedfinUpload}
                disabled={uploading || !redfinFile}
              >
                {uploading ? 'Uploading...' : 'Upload CSV'}
              </button>
            </div>
            {keyStatus.redfin && (
              <p style={{ marginTop: 12, color: 'var(--green)' }}>
                {keyStatus.redfin_count} listings loaded from Redfin CSV
              </p>
            )}
            {!keyStatus.redfin && !uploadMsg && (
              <p style={{ marginTop: 12, color: 'var(--text-muted)' }}>
                No Redfin data uploaded yet.
              </p>
            )}
            {uploadMsg && (
              <p style={{ marginTop: 12, color: uploadMsg.includes('Success') ? 'var(--green)' : 'var(--red)' }}>
                {uploadMsg}
              </p>
            )}
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        <div className="detail-section">
          <div className="section-header">API Keys</div>
          <div className="section-body">
            <div className="settings-form">
              {/* RentCast */}
              <div className="form-group">
                <label>
                  RentCast API Key
                  <span className={`source-indicator${keyStatus.rentcast ? ' connected' : ''}`} />
                </label>
                <input
                  type="password"
                  value={rentcastKey}
                  onChange={e => setRentcastKey(e.target.value)}
                  placeholder={keyStatus.rentcast ? '••••••• (configured)' : 'Enter your RentCast API key'}
                />
                <p className="help-text">
                  Get a free key at rentcast.io/api (50 calls/month). Provides for-sale listings.
                </p>
              </div>

              {/* RapidAPI */}
              <div className="form-group">
                <label>
                  RapidAPI Key
                  <span className={`source-indicator${keyStatus.rapidapi ? ' connected' : ''}`} />
                </label>
                <input
                  type="password"
                  value={rapidapiKey}
                  onChange={e => setRapidapiKey(e.target.value)}
                  placeholder={keyStatus.rapidapi ? '••••••• (configured)' : 'Enter your RapidAPI key'}
                />
                <p className="help-text">
                  Subscribe to "Realty in US" on rapidapi.com. Provides listings with photos.
                </p>
              </div>

              <button
                className="btn btn-primary"
                onClick={handleSaveKeys}
                disabled={saving || (!rentcastKey && !rapidapiKey)}
              >
                {saving ? 'Saving...' : 'Save Keys'}
              </button>

              {saveMsg && (
                <p style={{ marginTop: 12, color: saveMsg.includes('success') ? 'var(--green)' : 'var(--red)' }}>
                  {saveMsg}
                </p>
              )}
            </div>
          </div>
        </div>

        <div className="detail-section">
          <div className="section-header">Test Prediction Model</div>
          <div className="section-body">
            <div className="settings-form">
              <div className="form-group">
                <label>Purchase Price ($)</label>
                <input
                  type="text"
                  value={testForm.purchase_price}
                  onChange={e => setTestForm(f => ({ ...f, purchase_price: Number(e.target.value) }))}
                />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div className="form-group">
                  <label>Bedrooms</label>
                  <input
                    type="text"
                    value={testForm.bedrooms}
                    onChange={e => setTestForm(f => ({ ...f, bedrooms: Number(e.target.value) }))}
                  />
                </div>
                <div className="form-group">
                  <label>Bathrooms</label>
                  <input
                    type="text"
                    value={testForm.bathrooms}
                    onChange={e => setTestForm(f => ({ ...f, bathrooms: Number(e.target.value) }))}
                  />
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div className="form-group">
                  <label>Latitude</label>
                  <input
                    type="text"
                    value={testForm.latitude}
                    onChange={e => setTestForm(f => ({ ...f, latitude: Number(e.target.value) }))}
                  />
                </div>
                <div className="form-group">
                  <label>Longitude</label>
                  <input
                    type="text"
                    value={testForm.longitude}
                    onChange={e => setTestForm(f => ({ ...f, longitude: Number(e.target.value) }))}
                  />
                </div>
              </div>
              <button className="btn btn-primary" onClick={handleTest} disabled={testing}>
                {testing ? 'Predicting...' : 'Run Prediction'}
              </button>

              {testError && (
                <div className="error-message" style={{ marginTop: 16 }}>
                  <p>{testError}</p>
                </div>
              )}

              {testResult && (
                <div style={{ marginTop: 16 }}>
                  <div className="detail-stats">
                    <div className="detail-stat">
                      <span className="stat-label">STR Score</span>
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
                    <div className="detail-stat">
                      <span className="stat-label">ROI</span>
                      <span className="stat-value">{testResult.financials?.roi}%</span>
                    </div>
                    <div className="detail-stat">
                      <span className="stat-label">Monthly Cashflow</span>
                      <span className="stat-value" style={{
                        color: testResult.financials?.monthly_cashflow >= 0 ? 'var(--green)' : 'var(--red)'
                      }}>
                        {formatCurrency(testResult.financials?.monthly_cashflow)}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
