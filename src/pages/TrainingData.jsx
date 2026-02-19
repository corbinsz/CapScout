import { useMemo } from 'react';
import useProperties from '../hooks/useProperties';
import { useCity } from '../contexts/CityContext';
import CitySelector from '../components/layout/CitySelector';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import PropertyList from '../components/properties/PropertyList';
import { formatCurrency, formatPercent, formatNumber } from '../utils/formatters';

export default function TrainingData() {
  const { properties, total, page, setPage, loading, error, filters, updateFilters, reload } = useProperties({
    source: 'airbnb',
    min_price: null,
    max_price: null,
    sort_by: 'price',
    str_viable: false,
  });
  const { cityMeta } = useCity();
  const cityDisplay = cityMeta?.display || 'Austin, TX';

  const summary = useMemo(() => {
    if (!properties.length) return null;
    const avgRevenue = properties.reduce((s, p) => s + (p.monthly_revenue || 0), 0) / properties.length;
    const avgOcc = properties.reduce((s, p) => s + (p.occupancy_rate || 0), 0) / properties.length;
    const avgPrice = properties.reduce((s, p) => s + (p.price || 0), 0) / properties.length;
    return { avgRevenue, avgOcc, avgPrice };
  }, [properties]);

  const totalPages = Math.ceil(total / (filters.per_page || 20));

  return (
    <div>
      <div className="hero-section">
        <h1>Training Data</h1>
        <p className="training-data-note">
          These are the real Airbnb listings in {cityDisplay} that our ML model learns from.
          The model analyzes their nightly rates, occupancy, location, and property features
          to predict how a home for sale would perform as a short-term rental. The more
          listings near a for-sale property, the higher the prediction confidence.
        </p>
        {summary && (
          <div className="hero-stats">
            <div className="hero-stat">
              <span className="hero-stat-value">{formatCurrency(summary.avgPrice)}</span>
              <span className="hero-stat-label">Avg Nightly</span>
            </div>
            <div className="hero-stat">
              <span className="hero-stat-value">{formatCurrency(summary.avgRevenue)}</span>
              <span className="hero-stat-label">Avg Monthly Rev</span>
            </div>
            <div className="hero-stat">
              <span className="hero-stat-value">{formatPercent(summary.avgOcc)}</span>
              <span className="hero-stat-label">Avg Occupancy</span>
            </div>
            <div className="hero-stat">
              <span className="hero-stat-value">{formatNumber(total)}</span>
              <span className="hero-stat-label">Total Listings</span>
            </div>
          </div>
        )}
      </div>

      <div className="filter-panel">
        <CitySelector />
      </div>

      {error && <ErrorMessage message={error} onRetry={reload} />}
      {!error && (
        <div className={`property-results${loading ? ' results-loading' : ''}`}>
          {loading && <div className="results-overlay"><LoadingSpinner message="Loading listings..." /></div>}
          {properties.length > 0
            ? <PropertyList properties={properties} />
            : !loading && <p style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '40px 0' }}>No listings found.</p>
          }
        </div>
      )}

      {totalPages > 1 && (
        <div className="pagination">
          <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page <= 1}>Prev</button>
          <span className="page-indicator">Page {page} of {totalPages}</span>
          <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page >= totalPages}>Next</button>
        </div>
      )}
    </div>
  );
}
