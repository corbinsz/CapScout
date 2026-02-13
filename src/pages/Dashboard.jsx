import { useMemo } from 'react';
import useProperties from '../hooks/useProperties';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import FilterPanel from '../components/filters/FilterPanel';
import PropertyList from '../components/properties/PropertyList';
import { formatCurrency, formatPercent, formatNumber } from '../utils/formatters';

export default function Dashboard() {
  const { properties, total, page, setPage, loading, error, filters, updateFilters, reload, message } = useProperties();

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
        <h1>STR Investment Dashboard</h1>
        <p>
          {isForSale
            ? 'Austin, TX houses for sale — scored by predicted STR investment potential'
            : 'Austin, TX short-term rental listings ranked by predicted performance'}
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
              <span className="hero-stat-value">{formatNumber(total)}</span>
              <span className="hero-stat-label">Total Properties</span>
            </div>
          </div>
        )}
      </div>

      <FilterPanel
        filters={filters}
        onFilterChange={(key, val) => updateFilters({ [key]: val })}
        onReset={() => updateFilters({
          source: 'forsale', min_price: null, max_price: null, bedrooms: null,
          neighborhood: '', sort_by: 'score', sort_order: 'desc',
        })}
      />

      {message && <div className="fallback-banner">{message}</div>}
      {loading && <LoadingSpinner message="Loading properties..." />}
      {error && <ErrorMessage message={error} onRetry={reload} />}
      {!loading && !error && <PropertyList properties={properties} />}

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
