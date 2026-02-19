import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import Map, { Marker, Popup as MapPopup, NavigationControl } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { fetchProperty, fetchFeatureImportance } from '../services/propertyService';
import { useCity } from '../contexts/CityContext';
import { resolveBackendUrl } from '../services/api';
import ScoreBadge from '../components/properties/ScoreBadge';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import { formatCurrency, formatPercent } from '../utils/formatters';
import { getScoreColor } from '../utils/scoreHelpers';
import { scoreProperty as clientScore } from '../utils/scoringEngine';

const MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

const DEFAULT_ASSUMPTIONS = { mortgage_rate: 7, mortgage_term: 30, down_payment_pct: 20 };

const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json';

function CompMarker({ comp }) {
  const [show, setShow] = useState(false);
  return (
    <>
      <Marker longitude={comp.longitude} latitude={comp.latitude} anchor="center"
        onClick={(e) => { e.originalEvent.stopPropagation(); setShow(true); }}>
        <div className="map-dot map-dot-comp" />
      </Marker>
      {show && (
        <MapPopup longitude={comp.longitude} latitude={comp.latitude} anchor="bottom"
          onClose={() => setShow(false)} closeOnClick={false} className="map-popup">
          <div className="map-popup-body">
            <strong>{comp.name?.slice(0, 35) || `Listing #${comp.id}`}</strong>
            <div>{comp.bedrooms} bd · {formatCurrency(comp.price)}/night · {formatPercent(comp.occupancy_rate)} occ</div>
            <div style={{ color: '#64748b' }}>{comp.distance_km} km away</div>
            {comp.listing_url && (
              <a href={comp.listing_url} target="_blank" rel="noopener noreferrer"
                style={{ color: '#ff5a5f', fontWeight: 600, textDecoration: 'none', display: 'inline-block', marginTop: 4 }}>
                View on Airbnb &#8599;
              </a>
            )}
          </div>
        </MapPopup>
      )}
    </>
  );
}

function ScoreBreakdownBars({ breakdown }) {
  const bars = [
    { label: 'Gross Yield', value: breakdown.yield_score, max: 25 },
    { label: 'Cash-on-Cash ROI', value: breakdown.roi_score, max: 25 },
    { label: 'Cap Rate', value: breakdown.cap_rate_score, max: 20 },
    { label: 'DSCR', value: breakdown.dscr_score, max: 20 },
    { label: 'Occupancy', value: breakdown.occupancy_score, max: 10 },
  ];
  return (
    <div className="score-breakdown-bars">
      {bars.map(b => {
        const pct = (b.value / b.max) * 100;
        const color = pct >= 70 ? 'var(--green)' : pct >= 40 ? 'var(--yellow)' : 'var(--red)';
        return (
          <div key={b.label} className="sb-row">
            <span className="sb-label">{b.label}</span>
            <div className="sb-track">
              <div className="sb-fill" style={{ width: `${pct}%`, background: color }} />
            </div>
            <span className="sb-value">{b.value} / {b.max}</span>
          </div>
        );
      })}
    </div>
  );
}

function PropertyOverview({ property, pred, fin, neighborhood_stats, comparables, isForSale }) {
  // --- Property Features ---
  const features = [];
  if (property.property_type) features.push({ label: 'Type', value: property.property_type, color: 'orange' });
  features.push({ label: 'Bedrooms', value: property.bedrooms, color: 'orange' });
  features.push({ label: 'Bathrooms', value: property.bathrooms, color: 'orange' });
  if (property.sqft) features.push({ label: 'Sqft', value: property.sqft.toLocaleString(), color: 'teal' });
  if (property.year_built) features.push({ label: 'Year Built', value: property.year_built, color: 'teal' });
  if (property.lot_size) features.push({ label: 'Lot Size', value: `${property.lot_size.toLocaleString()} sqft`, color: 'teal' });
  if (property.price_per_sqft) features.push({ label: '$/Sqft', value: formatCurrency(property.price_per_sqft), color: 'indigo' });
  if (property.hoa_monthly) features.push({ label: 'HOA', value: `${formatCurrency(property.hoa_monthly)}/mo`, color: 'indigo' });
  if (property.days_on_market != null) features.push({ label: 'Days on Market', value: property.days_on_market, color: 'indigo' });
  if (!isForSale && property.accommodates) features.push({ label: 'Accommodates', value: `${property.accommodates} guests`, color: 'teal' });
  if (!isForSale && property.rating) features.push({ label: 'Guest Rating', value: `${property.rating.toFixed(1)} / 5`, color: 'orange' });

  // --- Area Insights (from real data) ---
  const areaFacts = [];
  if (neighborhood_stats) {
    areaFacts.push({ label: 'Active Listings', value: `${neighborhood_stats.count} in ${neighborhood_stats.neighbourhood_cleansed}`, color: 'orange' });
    areaFacts.push({ label: 'Avg Nightly Rate', value: formatCurrency(neighborhood_stats.avg_price), color: 'teal' });
    areaFacts.push({ label: 'Avg Occupancy', value: formatPercent(neighborhood_stats.avg_occupancy), color: 'teal' });
    areaFacts.push({ label: 'Avg Monthly Rev', value: formatCurrency(neighborhood_stats.avg_monthly_revenue), color: 'indigo' });
    if (neighborhood_stats.avg_rating) {
      areaFacts.push({ label: 'Avg Guest Rating', value: `${neighborhood_stats.avg_rating.toFixed(1)} / 5`, color: 'indigo' });
    }
  }
  if (pred.confidence) {
    areaFacts.push({ label: 'STR Nearby (5km)', value: `${pred.confidence.nearby_5km} listings`, color: 'orange' });
    if (pred.confidence.nearby_3km > 0) {
      areaFacts.push({ label: 'STR Within 3km', value: `${pred.confidence.nearby_3km} listings`, color: 'teal' });
    }
  }
  if (comparables?.length > 0) {
    const avgCompRate = comparables.reduce((s, c) => s + (c.price || 0), 0) / comparables.length;
    areaFacts.push({ label: 'Comp Avg Nightly', value: formatCurrency(avgCompRate), color: 'indigo' });
  }

  // --- Quick Insights (fun-fact pills) ---
  const insights = [];
  if (pred.monthly_breakdown?.length > 0) {
    const best = pred.monthly_breakdown.reduce((a, b) => b.monthly_revenue > a.monthly_revenue ? b : a);
    const worst = pred.monthly_breakdown.reduce((a, b) => b.monthly_revenue < a.monthly_revenue ? b : a);
    insights.push(`Peak season: ${MONTH_LABELS[best.month - 1]} (${formatCurrency(best.monthly_revenue)})`);
    insights.push(`Slow season: ${MONTH_LABELS[worst.month - 1]} (${formatCurrency(worst.monthly_revenue)})`);
    const totalDays = pred.monthly_breakdown.reduce((s, m) => {
      const d = new Date(2026, m.month, 0).getDate();
      return s + Math.round(m.occupancy_rate * d);
    }, 0);
    insights.push(`~${totalDays} booked days per year`);
  }
  if (property.year_built) {
    const age = new Date().getFullYear() - property.year_built;
    insights.push(`${age} year old property`);
  }
  if (property.days_on_market != null) {
    if (property.days_on_market <= 7) insights.push('New listing — just hit the market');
    else if (property.days_on_market >= 60) insights.push(`${property.days_on_market} days listed — may have room to negotiate`);
  }
  if (neighborhood_stats && pred.predicted_monthly_revenue) {
    const ratio = pred.predicted_monthly_revenue / neighborhood_stats.avg_monthly_revenue;
    if (ratio > 1.1) insights.push(`Predicted ${Math.round((ratio - 1) * 100)}% above neighborhood avg revenue`);
    else if (ratio < 0.9) insights.push(`Predicted ${Math.round((1 - ratio) * 100)}% below neighborhood avg revenue`);
    else insights.push('Revenue in line with neighborhood average');
  }

  return (
    <div className="overview-sections">
      <div className="overview-col">
        <div className="overview-col-header">Property Features</div>
        <div className="overview-grid">
          {features.map((f, i) => (
            <div key={i} className="overview-item">
              <span className={`overview-dot ${f.color}`} />
              <div className="overview-text">
                <span className="overview-label">{f.label}</span>
                <span className="overview-value">{f.value}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
      {areaFacts.length > 0 && (
        <div className="overview-col">
          <div className="overview-col-header">Area Insights</div>
          <div className="overview-grid">
            {areaFacts.map((f, i) => (
              <div key={i} className="overview-item">
                <span className={`overview-dot ${f.color}`} />
                <div className="overview-text">
                  <span className="overview-label">{f.label}</span>
                  <span className="overview-value">{f.value}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      {insights.length > 0 && (
        <div className="fun-facts">
          {insights.map((text, i) => (
            <span key={i} className="fun-fact">{text}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function CompCard({ comp }) {
  return (
    <div className="comp-card">
      <div className="comp-card-img">
        {comp.picture_url || comp.photo_url ? (
          <img src={resolveBackendUrl(comp.picture_url || comp.photo_url)} alt={comp.name || 'Listing'} loading="lazy" />
        ) : (
          <div className="comp-card-placeholder">No Image</div>
        )}
      </div>
      <div className="comp-card-body">
        <div className="comp-card-name">{comp.name?.slice(0, 50) || `Listing #${comp.id}`}</div>
        <div className="comp-card-specs">
          {comp.bedrooms} bd / {comp.bathrooms ?? '—'} ba
          <span className="spec-dot" />
          {comp.distance_km} km away
        </div>
        <div className="comp-card-stats">
          <div className="comp-stat">
            <span className="comp-stat-value">{formatCurrency(comp.price)}</span>
            <span className="comp-stat-label">/ night</span>
          </div>
          <div className="comp-stat">
            <span className="comp-stat-value">{formatPercent(comp.occupancy_rate)}</span>
            <span className="comp-stat-label">occupancy</span>
          </div>
          <div className="comp-stat">
            <span className="comp-stat-value">{formatCurrency(comp.monthly_revenue)}</span>
            <span className="comp-stat-label">/ month</span>
          </div>
        </div>
        {comp.rating != null && (
          <div className="comp-card-rating">{'★'.repeat(Math.round(comp.rating))}{'☆'.repeat(5 - Math.round(comp.rating))} {comp.rating.toFixed(1)}</div>
        )}
        {comp.listing_url && (
          <a href={comp.listing_url} target="_blank" rel="noopener noreferrer" className="comp-card-link">
            View on Airbnb &#8599;
          </a>
        )}
      </div>
    </div>
  );
}

function AssumptionsBar({ assumptions, onChange }) {
  return (
    <div className="assumptions-bar">
      <div className="assumption-field">
        <label>Down Payment %</label>
        <input
          type="number"
          min="0" max="100" step="1"
          value={assumptions.down_payment_pct}
          onChange={e => onChange({ ...assumptions, down_payment_pct: parseFloat(e.target.value) || 0 })}
        />
      </div>
      <div className="assumption-field">
        <label>Interest Rate %</label>
        <input
          type="number"
          min="0" max="20" step="0.125"
          value={assumptions.mortgage_rate}
          onChange={e => onChange({ ...assumptions, mortgage_rate: parseFloat(e.target.value) || 0 })}
        />
      </div>
      <div className="assumption-field">
        <label>Loan Term yr</label>
        <input
          type="number"
          min="1" max="50" step="1"
          value={assumptions.mortgage_term}
          onChange={e => onChange({ ...assumptions, mortgage_term: parseInt(e.target.value) || 30 })}
        />
      </div>
    </div>
  );
}

function FeatureImportance({ propertyId, cityKey }) {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleToggle = () => {
    if (!open && !data) {
      setLoading(true);
      fetchFeatureImportance(propertyId, cityKey)
        .then(d => setData(d))
        .catch(() => setData(null))
        .finally(() => setLoading(false));
    }
    setOpen(prev => !prev);
  };

  const maxImp = (features) => Math.max(...features.map(f => f.importance), 0.001);

  return (
    <div className="detail-section property-detail-full">
      <button className="section-header fi-toggle" onClick={handleToggle} style={{ cursor: 'pointer', width: '100%', border: 'none', textAlign: 'left' }}>
        Why This Score? {open ? '\u25B2' : '\u25BC'}
      </button>
      {open && (
        <div className="section-body">
          {loading && <div style={{ textAlign: 'center', padding: 20, color: 'var(--text-muted)' }}>Loading feature importance...</div>}
          {!loading && data && (
            <div className="fi-columns">
              {['price', 'occupancy'].map(key => data[key] && (
                <div key={key} className="fi-col">
                  <div className="fi-col-title">{data[key].label}</div>
                  {data[key].features.map((f, i) => (
                    <div key={i} className="fi-bar-row">
                      <span className="fi-bar-label">{f.name}</span>
                      <div className="fi-bar-track">
                        <div
                          className="fi-bar-fill"
                          style={{
                            width: `${(f.importance / maxImp(data[key].features)) * 100}%`,
                            background: key === 'price' ? 'var(--primary)' : 'var(--teal)',
                          }}
                        />
                      </div>
                      <span className="fi-bar-value">{(f.importance * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
          {!loading && !data && <div style={{ textAlign: 'center', padding: 20, color: 'var(--text-muted)' }}>Feature importance unavailable.</div>}
        </div>
      )}
    </div>
  );
}

function SensitivityAnalysis({ property, fin, pred }) {
  const [occAdj, setOccAdj] = useState(0);
  const [rateAdj, setRateAdj] = useState(0);
  const [intAdj, setIntAdj] = useState(0);

  const baseOcc = pred.predicted_occupancy_rate || 0;
  const baseNightly = pred.predicted_nightly_rate || 0;
  const baseMortgageRate = (fin.mortgage_rate || 7) / 100;

  const result = useMemo(() => {
    const adjOcc = Math.max(0, Math.min(1, baseOcc * (1 + occAdj / 100)));
    const adjNightly = baseNightly * (1 + rateAdj / 100);
    const adjMortgageRate = baseMortgageRate + intAdj / 100;
    const adjAnnualRev = adjNightly * adjOcc * 365;

    return clientScore({
      purchasePrice: property.price,
      annualRevenue: adjAnnualRev,
      occupancy: adjOcc,
      mortgageRate: Math.max(0, adjMortgageRate),
      mortgageTerm: fin.mortgage_term || 30,
      downPaymentPct: (fin.down_payment_pct || 20) / 100,
      taxRate: (fin.tax_rate || 1.8) / 100,
      insuranceRate: (fin.insurance_rate || 0.5) / 100,
      expenseRatio: (fin.expense_ratio || 35) / 100,
      hoaAnnual: fin.hoa_annual || 0,
      strPermitAnnual: fin.str_permit_annual || 0,
    });
  }, [occAdj, rateAdj, intAdj, baseOcc, baseNightly, baseMortgageRate, property.price, fin]);

  const currentScore = property.score || 0;
  const currentCashflow = fin.monthly_cashflow || 0;
  const currentCapRate = fin.cap_rate || 0;
  const scoreDelta = result.score - currentScore;
  const cashDelta = result.financials.monthlyCashflow - currentCashflow;
  const capDelta = result.financials.capRate - currentCapRate;

  const deltaColor = (val) => val > 0.01 ? 'var(--green)' : val < -0.01 ? 'var(--red)' : 'var(--text-muted)';
  const deltaSign = (val) => val > 0 ? '+' : '';

  return (
    <div className="detail-section property-detail-full">
      <div className="section-header">Sensitivity Analysis</div>
      <div className="section-body">
        <div className="sensitivity-sliders">
          <div className="sensitivity-slider">
            <label>Occupancy Adjustment</label>
            <input type="range" min={-30} max={20} step={1} value={occAdj}
              onChange={e => setOccAdj(Number(e.target.value))} />
            <span className="sensitivity-val">{occAdj > 0 ? '+' : ''}{occAdj}%</span>
          </div>
          <div className="sensitivity-slider">
            <label>Nightly Rate Adjustment</label>
            <input type="range" min={-30} max={30} step={1} value={rateAdj}
              onChange={e => setRateAdj(Number(e.target.value))} />
            <span className="sensitivity-val">{rateAdj > 0 ? '+' : ''}{rateAdj}%</span>
          </div>
          <div className="sensitivity-slider">
            <label>Interest Rate Adjustment</label>
            <input type="range" min={-2} max={3} step={0.125} value={intAdj}
              onChange={e => setIntAdj(Number(e.target.value))} />
            <span className="sensitivity-val">{intAdj > 0 ? '+' : ''}{intAdj}%</span>
          </div>
        </div>
        <div className="sensitivity-comparison">
          <div className="sensitivity-col">
            <div className="sensitivity-col-title">Current</div>
            <div className="sensitivity-metric">
              <span className="sensitivity-label">Score</span>
              <span className="sensitivity-value">{currentScore}</span>
            </div>
            <div className="sensitivity-metric">
              <span className="sensitivity-label">Monthly Cashflow</span>
              <span className="sensitivity-value">{formatCurrency(currentCashflow)}</span>
            </div>
            <div className="sensitivity-metric">
              <span className="sensitivity-label">Cap Rate</span>
              <span className="sensitivity-value">{currentCapRate}%</span>
            </div>
          </div>
          <div className="sensitivity-col sensitivity-col--adjusted">
            <div className="sensitivity-col-title">Adjusted</div>
            <div className="sensitivity-metric">
              <span className="sensitivity-label">Score</span>
              <span className="sensitivity-value">{result.score}
                <span className="sensitivity-delta" style={{ color: deltaColor(scoreDelta) }}>
                  {' '}{deltaSign(scoreDelta)}{scoreDelta.toFixed(1)}
                </span>
              </span>
            </div>
            <div className="sensitivity-metric">
              <span className="sensitivity-label">Monthly Cashflow</span>
              <span className="sensitivity-value">{formatCurrency(result.financials.monthlyCashflow)}
                <span className="sensitivity-delta" style={{ color: deltaColor(cashDelta) }}>
                  {' '}{deltaSign(cashDelta)}{formatCurrency(cashDelta)}
                </span>
              </span>
            </div>
            <div className="sensitivity-metric">
              <span className="sensitivity-label">Cap Rate</span>
              <span className="sensitivity-value">{result.financials.capRate}%
                <span className="sensitivity-delta" style={{ color: deltaColor(capDelta) }}>
                  {' '}{deltaSign(capDelta)}{capDelta.toFixed(2)}%
                </span>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function PropertyPage() {
  const { id } = useParams();
  const { currentCity } = useCity();
  const navigate = useNavigate();
  const initialCity = useRef(currentCity);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [assumptions, setAssumptions] = useState(DEFAULT_ASSUMPTIONS);
  const debounceRef = useRef(null);

  // If user switches cities while viewing a property, go back to dashboard
  useEffect(() => {
    if (currentCity !== initialCity.current) {
      navigate('/');
    }
  }, [currentCity, navigate]);

  // Initial fetch
  useEffect(() => {
    setLoading(true);
    fetchProperty(id, initialCity.current)
      .then(d => {
        setData(d);
        // Initialize assumptions from returned financials if available
        const fin = d?.property?.financials;
        if (fin) {
          setAssumptions({
            mortgage_rate: fin.mortgage_rate ?? DEFAULT_ASSUMPTIONS.mortgage_rate,
            mortgage_term: fin.mortgage_term ?? DEFAULT_ASSUMPTIONS.mortgage_term,
            down_payment_pct: fin.down_payment_pct ?? DEFAULT_ASSUMPTIONS.down_payment_pct,
          });
        }
      })
      .catch(err => setError(err.response?.data?.error || err.message))
      .finally(() => setLoading(false));
  }, [id]);

  // Debounced re-fetch when assumptions change from initial
  const handleAssumptionsChange = useCallback((newAssumptions) => {
    setAssumptions(newAssumptions);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      fetchProperty(id, initialCity.current, {
        mortgage_rate: newAssumptions.mortgage_rate,
        mortgage_term: newAssumptions.mortgage_term,
        down_payment_pct: newAssumptions.down_payment_pct,
      }).then(d => {
        setData(prev => ({
          ...prev,
          property: {
            ...prev.property,
            financials: d.property.financials,
            score: d.property.score,
            score_breakdown: d.property.score_breakdown,
          },
        }));
      }).catch(() => { /* keep existing data on error */ });
    }, 500);
  }, [id]);

  if (loading) return <LoadingSpinner message="Loading property..." />;
  if (error) return <ErrorMessage message={error} />;
  if (!data?.property) return <ErrorMessage message="Property not found" />;

  const { property, comparables, neighborhood_stats } = data;
  const fin = property.financials || {};
  const pred = property.predicted || {};
  const isForSale = property.source === 'rentcast' || property.source === 'realtor' || property.source === 'redfin';

  return (
    <div>
      {/* Breadcrumb */}
      <div className="breadcrumb">
        <Link to="/">Dashboard</Link>
        <span className="breadcrumb-sep">/</span>
        <span>{property.address || property.neighbourhood || 'Property'}</span>
        <span className="breadcrumb-sep">/</span>
        <span>{property.name || `#${property.id}`}</span>
      </div>

      {/* Hero Card */}
      <div className="property-hero">
        {/* Hero photo if available */}
        {property.photo_url && (
          <div className="property-hero-photo">
            <img src={resolveBackendUrl(property.photo_url)} alt={property.address || property.name || 'Property'} />
          </div>
        )}

        <div className="property-hero-top">
          <div className="property-hero-info">
            <h1>{property.address || property.name || `Property #${property.id}`}</h1>
            <p className="hero-meta">
              {property.neighbourhood && `${property.neighbourhood} · `}
              {property.bedrooms} bd / {property.bathrooms} ba
              {!isForSale && property.accommodates ? ` · Sleeps ${property.accommodates}` : ''}
              {isForSale && property.sqft ? ` · ${property.sqft.toLocaleString()} sqft` : ''}
              {property.source && (
                <span className="hero-source-tag"> · {property.source === 'redfin' ? 'Redfin' : property.source === 'rentcast' ? 'RentCast' : property.source === 'realtor' ? 'Realtor.com' : 'Airbnb'}</span>
              )}
            </p>
            {(property.url || property.listing_url) && (
              <a href={property.url || property.listing_url} target="_blank" rel="noopener noreferrer" className="hero-external-link">
                View on {property.source === 'redfin' ? 'Redfin' : property.source === 'airbnb' || property.listing_url ? 'Airbnb' : 'Listing Site'} &#8599;
              </a>
            )}
          </div>
          {isForSale && (
            <div className="property-hero-score">
              <ScoreBadge score={property.score} variant="bar" />
            </div>
          )}
        </div>

        {/* Key Metrics Strip */}
        <div className="key-metrics-strip">
          {isForSale ? (
            <>
              <div className="kms-item">
                <span className="kms-value">{formatCurrency(property.price)}</span>
                <span className="kms-label">Purchase Price</span>
              </div>
              <div className="kms-item">
                <span className="kms-value">{formatCurrency(pred.predicted_nightly_rate)}</span>
                <span className="kms-label">Pred. Nightly</span>
              </div>
              <div className="kms-item">
                <span className="kms-value">{formatCurrency(pred.predicted_monthly_revenue)}</span>
                <span className="kms-label">Pred. Monthly Rev</span>
              </div>
              <div className="kms-item">
                <span className="kms-value">{formatPercent(pred.predicted_occupancy_rate)}</span>
                <span className="kms-label">Pred. Occupancy</span>
              </div>
            </>
          ) : (
            <>
              <div className="kms-item">
                <span className="kms-value">{formatCurrency(property.price)}</span>
                <span className="kms-label">Nightly Rate</span>
              </div>
              <div className="kms-item">
                <span className="kms-value">{formatPercent(property.occupancy_rate)}</span>
                <span className="kms-label">Occupancy</span>
              </div>
              <div className="kms-item">
                <span className="kms-value">{formatCurrency(property.monthly_revenue)}</span>
                <span className="kms-label">Monthly Revenue</span>
              </div>
              <div className="kms-item">
                <span className="kms-value">{property.rating?.toFixed(2) ?? '—'}</span>
                <span className="kms-label">Rating</span>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="property-detail">
        {/* Property Overview */}
        <div className="detail-section property-detail-full">
          <div className="section-header">
            <div>Property Overview</div>
            <div className="section-subtitle">Key details and insights at a glance</div>
          </div>
          <div className="section-body">
            <PropertyOverview
              property={property}
              pred={pred}
              fin={fin}
              neighborhood_stats={neighborhood_stats}
              comparables={comparables}
              isForSale={isForSale}
            />
          </div>
        </div>

        {/* Score Breakdown + ML Predictions — combined */}
        {(isForSale && property.score_breakdown) || pred.predicted_nightly_rate ? (
          <div className="detail-section property-detail-full">
            <div className="section-header" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              {isForSale ? 'Score Breakdown & Predictions' : 'ML Predictions'}
              {pred.confidence && (
                <span className={`confidence-badge confidence-${pred.confidence.level}`}>
                  {pred.confidence.level === 'high' ? 'High' : pred.confidence.level === 'medium' ? 'Medium' : 'Low'} Confidence
                  <span className="confidence-detail"> ({pred.confidence.score}%)</span>
                </span>
              )}
            </div>
            <div className="section-body">
              {pred.confidence?.level === 'low' && (
                <div className="confidence-warning">
                  Few training listings nearby — only {pred.confidence.nearby_5km} Airbnb listings within 5 km.
                  Predictions in this area may be less reliable.
                  {pred.confidence.out_of_range_features?.map(f => (
                    <span key={f.feature}> {f.detail}</span>
                  ))}
                </div>
              )}
              <div className="combined-score-pred">
                {isForSale && property.score_breakdown && (
                  <div className="combined-left">
                    <ScoreBreakdownBars breakdown={property.score_breakdown} />
                  </div>
                )}
                <div className="combined-right">
                  <div className="detail-stats">
                    <div className="detail-stat">
                      <span className="stat-label">Pred. Nightly</span>
                      <span className="stat-value">{formatCurrency(pred.predicted_nightly_rate)}</span>
                      {pred.prediction_range && (
                        <span className="stat-range">{formatCurrency(pred.prediction_range.nightly_low)} – {formatCurrency(pred.prediction_range.nightly_high)}</span>
                      )}
                    </div>
                    <div className="detail-stat">
                      <span className="stat-label">Pred. Occupancy</span>
                      <span className="stat-value">{formatPercent(pred.predicted_occupancy_rate)}</span>
                      {pred.prediction_range && (
                        <span className="stat-range">{formatPercent(pred.prediction_range.occupancy_low)} – {formatPercent(pred.prediction_range.occupancy_high)}</span>
                      )}
                    </div>
                    <div className="detail-stat">
                      <span className="stat-label">Pred. Monthly</span>
                      <span className="stat-value">{formatCurrency(pred.predicted_monthly_revenue)}</span>
                    </div>
                    <div className="detail-stat">
                      <span className="stat-label">Pred. Annual</span>
                      <span className="stat-value">{formatCurrency(pred.predicted_annual_revenue)}</span>
                      {pred.prediction_range && (
                        <span className="stat-range">{formatCurrency(pred.prediction_range.annual_revenue_low)} – {formatCurrency(pred.prediction_range.annual_revenue_high)}</span>
                      )}
                    </div>
                  </div>
                  {pred.confidence && (
                    <div className="confidence-footer">
                      Based on {pred.confidence.n_training?.toLocaleString()} training listings ·
                      {' '}{pred.confidence.nearby_5km} within 5 km · {pred.confidence.nearby_3km} within 3 km
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : null}

        {/* Feature Importance — lazy-loads on click */}
        {isForSale && (
          <FeatureImportance propertyId={property.id} cityKey={initialCity.current} />
        )}

        {/* Financial Analysis — only for properties with a real purchase price */}
        {isForSale && fin.cap_rate != null && (
          <div className="detail-section property-detail-full">
            <div className="section-header">Financial Analysis</div>
            <AssumptionsBar assumptions={assumptions} onChange={handleAssumptionsChange} />
            <div className="section-body">
              <table className="income-statement">
                <tbody>
                  {/* --- Revenue --- */}
                  <tr className="is-section-header"><td colSpan="3">Revenue</td></tr>
                  <tr>
                    <td className="is-label">Gross Rental Income</td>
                    <td className="is-annual">{formatCurrency(fin.annual_revenue)}</td>
                    <td className="is-monthly">{formatCurrency(fin.annual_revenue / 12)}/mo</td>
                  </tr>

                  {/* --- Operating Expenses (research-backed splits of total OpEx) --- */}
                  <tr className="is-section-header"><td colSpan="3">Operating Expenses</td></tr>
                  <tr>
                    <td className="is-label">Cleaning &amp; Turnover</td>
                    <td className="is-annual is-negative">({formatCurrency(fin.operating_expenses * 0.30)})</td>
                    <td className="is-monthly is-negative">({formatCurrency(fin.operating_expenses * 0.30 / 12)})/mo</td>
                  </tr>
                  <tr>
                    <td className="is-label">Utilities (Electric, Water, Internet)</td>
                    <td className="is-annual is-negative">({formatCurrency(fin.operating_expenses * 0.20)})</td>
                    <td className="is-monthly is-negative">({formatCurrency(fin.operating_expenses * 0.20 / 12)})/mo</td>
                  </tr>
                  <tr>
                    <td className="is-label">Maintenance &amp; Repairs</td>
                    <td className="is-annual is-negative">({formatCurrency(fin.operating_expenses * 0.18)})</td>
                    <td className="is-monthly is-negative">({formatCurrency(fin.operating_expenses * 0.18 / 12)})/mo</td>
                  </tr>
                  <tr>
                    <td className="is-label">Supplies, Linens &amp; Consumables</td>
                    <td className="is-annual is-negative">({formatCurrency(fin.operating_expenses * 0.12)})</td>
                    <td className="is-monthly is-negative">({formatCurrency(fin.operating_expenses * 0.12 / 12)})/mo</td>
                  </tr>
                  <tr>
                    <td className="is-label">Platform Fees &amp; Software (~3% of rev)</td>
                    <td className="is-annual is-negative">({formatCurrency(fin.operating_expenses * 0.12)})</td>
                    <td className="is-monthly is-negative">({formatCurrency(fin.operating_expenses * 0.12 / 12)})/mo</td>
                  </tr>
                  <tr>
                    <td className="is-label">Misc &amp; Reserves</td>
                    <td className="is-annual is-negative">({formatCurrency(fin.operating_expenses * 0.08)})</td>
                    <td className="is-monthly is-negative">({formatCurrency(fin.operating_expenses * 0.08 / 12)})/mo</td>
                  </tr>
                  <tr className="is-subtotal-row">
                    <td className="is-label">Total Operating Expenses ({fin.expense_ratio}%)</td>
                    <td className="is-annual is-negative">({formatCurrency(fin.operating_expenses)})</td>
                    <td className="is-monthly is-negative">({formatCurrency(fin.operating_expenses / 12)})/mo</td>
                  </tr>

                  {/* --- NOI --- */}
                  <tr className="is-total-row">
                    <td className="is-label">Net Operating Income (NOI)</td>
                    <td className="is-annual">{formatCurrency(fin.noi)}</td>
                    <td className="is-monthly">{formatCurrency(fin.noi / 12)}/mo</td>
                  </tr>

                  {/* --- Fixed Costs --- */}
                  <tr className="is-section-header"><td colSpan="3">Debt Service &amp; Fixed Costs</td></tr>
                  <tr>
                    <td className="is-label">Mortgage ({assumptions.mortgage_rate}% / {assumptions.mortgage_term}yr, {assumptions.down_payment_pct}% down)</td>
                    <td className="is-annual is-negative">({formatCurrency(fin.annual_mortgage)})</td>
                    <td className="is-monthly is-negative">({formatCurrency(fin.annual_mortgage / 12)})/mo</td>
                  </tr>
                  <tr>
                    <td className="is-label">Property Tax ({fin.tax_rate}%)</td>
                    <td className="is-annual is-negative">({formatCurrency(fin.annual_taxes)})</td>
                    <td className="is-monthly is-negative">({formatCurrency(fin.annual_taxes / 12)})/mo</td>
                  </tr>
                  <tr>
                    <td className="is-label">Insurance ({fin.insurance_rate}%)</td>
                    <td className="is-annual is-negative">({formatCurrency(fin.annual_insurance)})</td>
                    <td className="is-monthly is-negative">({formatCurrency(fin.annual_insurance / 12)})/mo</td>
                  </tr>
                  {fin.hoa_annual > 0 && (
                    <tr>
                      <td className="is-label">HOA Fees</td>
                      <td className="is-annual is-negative">({formatCurrency(fin.hoa_annual)})</td>
                      <td className="is-monthly is-negative">({formatCurrency(fin.hoa_annual / 12)})/mo</td>
                    </tr>
                  )}
                  {fin.str_permit_annual > 0 && (
                    <tr>
                      <td className="is-label">STR Permit</td>
                      <td className="is-annual is-negative">({formatCurrency(fin.str_permit_annual)})</td>
                      <td className="is-monthly is-negative">({formatCurrency(fin.str_permit_annual / 12)})/mo</td>
                    </tr>
                  )}

                  {/* --- Net Cashflow --- */}
                  <tr className={`is-total-row is-bottom-line ${fin.annual_cashflow >= 0 ? 'is-positive' : 'is-loss'}`}>
                    <td className="is-label">Net Cashflow</td>
                    <td className="is-annual">{formatCurrency(fin.annual_cashflow)}</td>
                    <td className="is-monthly">{formatCurrency(fin.monthly_cashflow)}/mo</td>
                  </tr>
                </tbody>
              </table>

              {/* --- Return Metrics --- */}
              <div className="is-returns">
                <div className="is-return-card">
                  <span className="is-return-value">{fin.cap_rate}%</span>
                  <span className="is-return-label">Cap Rate</span>
                </div>
                <div className="is-return-card">
                  <span className="is-return-value">{fin.roi}%</span>
                  <span className="is-return-label">Cash-on-Cash ROI</span>
                </div>
                <div className="is-return-card">
                  <span className="is-return-value">{fin.dscr}</span>
                  <span className="is-return-label">DSCR</span>
                </div>
                <div className="is-return-card">
                  <span className="is-return-value">{fin.revenue_to_price}%</span>
                  <span className="is-return-label">Gross Yield</span>
                </div>
                <div className="is-return-card">
                  <span className="is-return-value">{formatCurrency(fin.down_payment)}</span>
                  <span className="is-return-label">Down Payment</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Sensitivity Analysis — client-side, no API calls */}
        {isForSale && fin.cap_rate != null && (
          <SensitivityAnalysis property={property} fin={fin} pred={pred} />
        )}

        {/* Monthly Forecast Chart + Table */}
        {pred.monthly_breakdown?.length > 0 && (
          <div className="detail-section property-detail-full">
            <div className="section-header">Monthly Forecast</div>
            <div className="section-body">
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={pred.monthly_breakdown.map(m => ({
                  ...m,
                  month_label: MONTH_LABELS[m.month - 1],
                  occupancy_pct: Math.round(m.occupancy_rate * 100),
                }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e4e6ed" />
                  <XAxis dataKey="month_label" tick={{ fill: '#3d3d50', fontSize: 12 }} />
                  <YAxis yAxisId="left" tick={{ fill: '#3d3d50', fontSize: 12 }} tickFormatter={v => `$${v}`} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fill: '#3d3d50', fontSize: 12 }} tickFormatter={v => `${v}%`} domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{
                      background: '#ffffff',
                      border: '1px solid #e4e6ed',
                      borderRadius: 12,
                      boxShadow: '0 6px 20px rgba(0,0,0,0.08)',
                      padding: '10px 14px',
                    }}
                    labelStyle={{ fontWeight: 700, color: '#1a1a2e', marginBottom: 4 }}
                    itemStyle={{ fontSize: 13, padding: '2px 0' }}
                    formatter={(value, name) => {
                      if (name === 'Nightly Rate') return [`$${value}`, name];
                      if (name === 'Occupancy') return [`${value}%`, name];
                      if (name === 'Revenue') return [`$${value.toLocaleString()}`, name];
                      return [value, name];
                    }}
                  />
                  <Legend />
                  <Bar yAxisId="left" dataKey="nightly_rate" name="Nightly Rate" fill="#BF5700" radius={[3, 3, 0, 0]} />
                  <Bar yAxisId="left" dataKey="monthly_revenue" name="Revenue" fill="#0d9488" radius={[3, 3, 0, 0]} />
                  <Bar yAxisId="right" dataKey="occupancy_pct" name="Occupancy" fill="#4f46e5" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Location + Neighborhood Stats — combined */}
        {property.latitude != null && property.longitude != null && (
          <div className="detail-section property-detail-full">
            <div className="section-header">
              Location{neighborhood_stats ? ` — ${neighborhood_stats.neighbourhood_cleansed}` : ''}
            </div>
            <div style={{ height: 300, position: 'relative' }}>
              <Map
                key={`map-${property.id}`}
                initialViewState={{
                  longitude: Number(property.longitude),
                  latitude: Number(property.latitude),
                  zoom: 14,
                }}
                style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
                mapStyle={MAP_STYLE}
                scrollZoom={false}
              >
                <NavigationControl position="top-right" showCompass={false} />
                <Marker longitude={Number(property.longitude)} latitude={Number(property.latitude)} anchor="center">
                  <div className="map-dot map-dot-lg" style={{ background: getScoreColor(property.score) }} />
                </Marker>
                {comparables?.map(c => (
                  <CompMarker key={c.id} comp={c} />
                ))}
              </Map>
            </div>
            {neighborhood_stats && (
              <div className="section-body neighborhood-bar">
                <div className="detail-stats">
                  <div className="detail-stat">
                    <span className="stat-label">Listings Nearby</span>
                    <span className="stat-value">{neighborhood_stats.count}</span>
                  </div>
                  <div className="detail-stat">
                    <span className="stat-label">Avg Nightly</span>
                    <span className="stat-value">{formatCurrency(neighborhood_stats.avg_price)}</span>
                  </div>
                  <div className="detail-stat">
                    <span className="stat-label">Avg Occupancy</span>
                    <span className="stat-value">{formatPercent(neighborhood_stats.avg_occupancy)}</span>
                  </div>
                  <div className="detail-stat">
                    <span className="stat-label">Avg Monthly Rev</span>
                    <span className="stat-value">{formatCurrency(neighborhood_stats.avg_monthly_revenue)}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Comparable Listings — Card Grid */}
        {comparables?.length > 0 && (
          <div className="detail-section property-detail-full">
            <div className="section-header">
              <div>Nearby Airbnb Listings</div>
              <div className="section-subtitle">Comparable short-term rentals in the area</div>
            </div>
            <div className="section-body">
              <div className="comp-card-grid">
                {comparables.map(c => <CompCard key={c.id} comp={c} />)}
              </div>
            </div>
          </div>
        )}

        {/* Metrics Key */}
        <div className="detail-section property-detail-full">
          <div className="section-header">
            <div>Understanding These Metrics</div>
            <div className="section-subtitle">What the numbers mean and how they're calculated</div>
          </div>
          <div className="section-body">
            <div className="metrics-key">
              {isForSale && (
                <>
                  <div className="metrics-key-item">
                    <span className="metrics-key-term">Investment Score (0-100)</span>
                    <span className="metrics-key-desc">Composite rating based on industry-standard STR underwriting metrics: Gross Yield (25 pts), Cash-on-Cash ROI (25 pts), Cap Rate (20 pts), DSCR (20 pts), and Occupancy (10 pts). 75+ is excellent, 55-74 is good, 40-54 is fair, below 40 is poor.</span>
                  </div>
                  <div className="metrics-key-item">
                    <span className="metrics-key-term">Gross Yield (Revenue / Price)</span>
                    <span className="metrics-key-desc">Annual rental revenue divided by purchase price. The #1 quick-screen metric per BiggerPockets and AirDNA. 10% is market baseline, 15% is a good deal, 20%+ is excellent.</span>
                  </div>
                  <div className="metrics-key-item">
                    <span className="metrics-key-term">Cash-on-Cash ROI</span>
                    <span className="metrics-key-desc">Annual cashflow divided by your down payment. The primary "deal or no deal" metric for leveraged investors. 8% is good, 15% is excellent, 20%+ is exceptional.</span>
                  </div>
                  <div className="metrics-key-item">
                    <span className="metrics-key-term">Cap Rate</span>
                    <span className="metrics-key-desc">Net Operating Income (NOI) divided by purchase price. Measures property profitability before financing. STR cap rates typically run 1.5-2x higher than long-term rentals. 5% is decent, 8% is good, 10%+ is excellent.</span>
                  </div>
                  <div className="metrics-key-item">
                    <span className="metrics-key-term">DSCR (Debt Service Coverage Ratio)</span>
                    <span className="metrics-key-desc">NOI divided by total debt service (mortgage + taxes + insurance + HOA + permits). Critical for leveraged purchases — lenders typically want 1.25+. Below 1.0 means the property doesn't cover its debt from rental income alone.</span>
                  </div>
                  <div className="metrics-key-item">
                    <span className="metrics-key-term">NOI (Net Operating Income)</span>
                    <span className="metrics-key-desc">Annual rental revenue minus operating expenses (cleaning, maintenance, supplies, management, platform fees). Does not include mortgage, taxes, insurance, or HOA.</span>
                  </div>
                  <div className="metrics-key-item">
                    <span className="metrics-key-term">Operating Expenses</span>
                    <span className="metrics-key-desc">Estimated at {fin.expense_ratio || 35}% of gross revenue for a self-managed STR. Breakdown based on industry data (AirDNA, BiggerPockets): cleaning & turnover (~30%), utilities (~20%), maintenance & repairs (~18%), supplies & linens (~12%), platform fees & software (~12%), misc & reserves (~8%). Professionally managed properties run 40-50%+.</span>
                  </div>
                </>
              )}
              <div className="metrics-key-item">
                <span className="metrics-key-term">Confidence Level</span>
                <span className="metrics-key-desc">How reliable our predictions are for this location. Based on the density of nearby Airbnb listings in our training data, weighted by distance (1km, 3km, 5km). More nearby listings means higher confidence. Low confidence areas may have less accurate predictions.</span>
              </div>
              <div className="metrics-key-item">
                <span className="metrics-key-term">Predicted Occupancy</span>
                <span className="metrics-key-desc">Estimated percentage of nights booked per year, based on comparable Airbnb listings nearby. Varies seasonally — peak months may hit 80%+ while slow months drop to 40-50%.</span>
              </div>
              <div className="metrics-key-item">
                <span className="metrics-key-term">Predicted Nightly Rate</span>
                <span className="metrics-key-desc">Average nightly price our model estimates for this property, based on bedrooms, bathrooms, location, and comparable listings. The monthly chart shows seasonal variation.</span>
              </div>
              {isForSale && (
                <div className="metrics-key-item">
                  <span className="metrics-key-term">Monthly Cashflow</span>
                  <span className="metrics-key-desc">What you pocket (or lose) each month after all expenses: revenue minus operating costs, mortgage, taxes, insurance, HOA, and permits. Green = positive, red = you're paying out of pocket.</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
