import { useMemo } from 'react';
import useProperties from '../hooks/useProperties';
import { useCity } from '../contexts/CityContext';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import FilterPanel from '../components/filters/FilterPanel';
import PropertyList from '../components/properties/PropertyList';
import { formatCurrency, formatPercent, formatNumber } from '../utils/formatters';

export default function Dashboard() {
  const { properties, total, totalAvailable, page, setPage, loading, error, filters, updateFilters, reload, message } = useProperties();
  const { cityMeta } = useCity();

  const cityDisplay = cityMeta?.display || 'Austin, TX';
  const isForSale = (filters.source || 'forsale') !== 'airbnb';

  const summary = useMemo(() => {
    if (!properties.length) return null;
    const scored = properties.filter(p => p.score != null);
    const avgScore = scored.length ? scored.reduce((s, p) => s + p.score, 0) / scored.length : 0;
    const avgRevenue = properties.reduce((s, p) => s + (p.predicted?.predicted_monthly_revenue || p.monthly_revenue || 0), 0) / properties.length;
    const avgOcc = properties.reduce((s, p) => s + (p.predicted?.predicted_occupancy_rate || p.occupancy_rate || 0), 0) / properties.length;
    const avgPrice = properties.reduce((s, p) => s + (p.price || 0), 0) / properties.length;
    return { avgScore, avgRevenue, avgOcc, avgPrice };
  }, [properties]);

  const totalPages = Math.ceil(total / (filters.per_page || 20));

  return (
    <div>
      <div className="hero-section">
        <h1>CapScout Dashboard</h1>
        <p>
          {isForSale
            ? `${cityDisplay} houses for sale — scored by short-term rental investment potential`
            : `${cityDisplay} active Airbnb listings — this is the real-world data our ML model trains on to generate predictions for homes for sale`}
        </p>
        {summary && (
          <div className="hero-stats">
            <div className="hero-stat">
              <span className="hero-stat-value">{summary.avgScore.toFixed(1)}</span>
              <span className="hero-stat-label">Avg Score</span>
            </div>
            <div className="hero-stat">
              <span className="hero-stat-value">{formatCurrency(summary.avgPrice)}</span>
              <span className="hero-stat-label">{isForSale ? 'Avg Price' : 'Avg Nightly'}</span>
            </div>
            <div className="hero-stat">
              <span className="hero-stat-value">{formatCurrency(summary.avgRevenue)}</span>
              <span className="hero-stat-label">{isForSale ? 'Avg Pred. Rev' : 'Avg Monthly Rev'}</span>
            </div>
            <div className="hero-stat">
              <span className="hero-stat-value">{formatPercent(summary.avgOcc)}</span>
              <span className="hero-stat-label">{isForSale ? 'Avg Pred. Occ' : 'Avg Occupancy'}</span>
            </div>
            <div className="hero-stat">
              <span className="hero-stat-value">{formatNumber(totalAvailable)}</span>
              <span className="hero-stat-label">Total Properties</span>
            </div>
          </div>
        )}
      </div>

      <FilterPanel
        filters={filters}
        onFilterChange={(key, val, extra) => updateFilters({ [key]: val, ...extra })}
        onReset={() => updateFilters({
          source: 'forsale', min_price: null, max_price: null, bedrooms: null,
          neighborhood: '', sort_by: 'score', sort_order: 'desc',
          mortgage_rate: null, mortgage_term: null, down_payment_pct: null,
        })}
      />

      {message && <div className="fallback-banner">{message}</div>}
      {error && <ErrorMessage message={error} onRetry={reload} />}
      {!error && (
        <div className={`property-results${loading ? ' results-loading' : ''}`}>
          {loading && <div className="results-overlay"><LoadingSpinner message="Updating..." /></div>}
          {properties.length > 0
            ? <PropertyList properties={properties} />
            : !loading && <p style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '40px 0' }}>No properties match your filters.</p>
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
      {filters.sort_by === 'score' && totalAvailable > total && (
        <p className="score-note">
          Showing top {formatNumber(total)} investment candidates of {formatNumber(totalAvailable)} listings.
          Sort by price to browse all properties.
        </p>
      )}
    </div>
  );
}
