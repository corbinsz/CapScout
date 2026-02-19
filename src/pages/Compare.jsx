import { useState, useEffect } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { fetchProperty } from '../services/propertyService';
import { resolveBackendUrl } from '../services/api';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import { formatCurrency, formatPercent } from '../utils/formatters';

export default function Compare() {
  const [searchParams] = useSearchParams();
  const ids = (searchParams.get('ids') || '').split(',').filter(Boolean);
  const city = searchParams.get('city') || 'austin';
  const [properties, setProperties] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!ids.length) { setLoading(false); return; }
    setLoading(true);
    Promise.all(ids.map(id => fetchProperty(id, city)))
      .then(results => setProperties(results.map(r => r.property).filter(Boolean)))
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, [searchParams]);

  if (loading) return <LoadingSpinner message="Loading comparison..." />;
  if (error) return <ErrorMessage message={error} />;
  if (!properties.length) return (
    <div className="empty-state">
      <span className="empty-state-icon">&#8644;</span>
      <h3>No properties to compare</h3>
      <p>Select 2-3 properties from the <Link to="/">Dashboard</Link> to compare them side by side.</p>
    </div>
  );

  const rows = [
    { label: 'Purchase Price', fn: p => formatCurrency(p.price) },
    { label: 'Bedrooms', fn: p => p.bedrooms },
    { label: 'Bathrooms', fn: p => p.bathrooms },
    { label: 'Sqft', fn: p => p.sqft ? p.sqft.toLocaleString() : '--' },
    { label: 'Year Built', fn: p => p.year_built || '--' },
    { label: 'Score', fn: p => p.score != null ? p.score : '--', highlight: true },
    { label: 'Pred. Nightly Rate', fn: p => formatCurrency(p.predicted?.predicted_nightly_rate) },
    { label: 'Pred. Occupancy', fn: p => formatPercent(p.predicted?.predicted_occupancy_rate) },
    { label: 'Pred. Monthly Rev', fn: p => formatCurrency(p.predicted?.predicted_monthly_revenue) },
    { label: 'Pred. Annual Rev', fn: p => formatCurrency(p.predicted?.predicted_annual_revenue) },
    { label: 'Cap Rate', fn: p => p.financials?.cap_rate != null ? `${p.financials.cap_rate}%` : '--' },
    { label: 'Cash-on-Cash ROI', fn: p => p.financials?.roi != null ? `${p.financials.roi}%` : '--' },
    { label: 'DSCR', fn: p => p.financials?.dscr ?? '--' },
    { label: 'Gross Yield', fn: p => p.financials?.revenue_to_price != null ? `${p.financials.revenue_to_price}%` : '--' },
    { label: 'Monthly Cashflow', fn: p => formatCurrency(p.financials?.monthly_cashflow), color: true },
    { label: 'Down Payment', fn: p => formatCurrency(p.financials?.down_payment) },
    { label: 'NOI', fn: p => formatCurrency(p.financials?.noi) },
  ];

  return (
    <div>
      <div className="breadcrumb">
        <Link to="/">Dashboard</Link>
        <span className="breadcrumb-sep">/</span>
        <span>Compare</span>
      </div>

      <div className="hero-section">
        <h1>Property Comparison</h1>
        <p>Side-by-side comparison of {properties.length} properties</p>
      </div>

      <div className="compare-table-wrap">
        <table className="compare-table">
          <thead>
            <tr>
              <th className="compare-label-col"></th>
              {properties.map(p => (
                <th key={p.id} className="compare-prop-header">
                  {p.photo_url && (
                    <img src={resolveBackendUrl(p.photo_url)} alt="" className="compare-thumb" />
                  )}
                  <Link to={`/property/${p.id}`} className="compare-prop-name">
                    {p.address || p.name || `Property #${p.id}`}
                  </Link>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(row => (
              <tr key={row.label} className={row.highlight ? 'compare-highlight-row' : ''}>
                <td className="compare-label">{row.label}</td>
                {properties.map(p => {
                  const val = row.fn(p);
                  let className = 'compare-value';
                  if (row.color && p.financials?.monthly_cashflow < 0) className += ' text-negative';
                  return <td key={p.id} className={className}>{val}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
