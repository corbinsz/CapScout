import { useMemo, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import useProperties from '../hooks/useProperties';
import { useCity } from '../contexts/CityContext';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import FilterPanel from '../components/filters/FilterPanel';
import PropertyList from '../components/properties/PropertyList';
import { formatCurrency, formatPercent, formatNumber } from '../utils/formatters';

export default function Dashboard() {
  const { properties, total, totalAvailable, page, setPage, loading, error, filters, updateFilters, reload, message } = useProperties();
  const { cityMeta, currentCity } = useCity();
  const navigate = useNavigate();
  const [compareIds, setCompareIds] = useState(new Set());

  const handleCompareToggle = useCallback((id) => {
    setCompareIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else if (next.size < 3) {
        next.add(id);
      }
      return next;
    });
  }, []);

  const goCompare = () => {
    const ids = [...compareIds].join(',');
    navigate(`/compare?ids=${ids}&city=${currentCity}`);
  };

  const cityDisplay = cityMeta?.display || 'Austin, TX';

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
        <p>{cityDisplay} homes for sale -- scored by short-term rental investment potential.</p>
        <p className="hero-detail">
          Each property is run through our ML model to predict nightly rate and occupancy,
          then scored on five financial metrics. Use the filters below to narrow by price,
          bedrooms, or financing assumptions. Adjust the interest rate, down payment, and
          loan term to see how different scenarios affect each property's investment score.
        </p>
        {summary && (
          <div className="hero-stats">
            <div className="hero-stat">
              <span className="hero-stat-value">{summary.avgScore.toFixed(1)}</span>
              <span className="hero-stat-label">Avg Score</span>
            </div>
            <div className="hero-stat">
              <span className="hero-stat-value">{formatCurrency(summary.avgPrice)}</span>
              <span className="hero-stat-label">Avg Price</span>
            </div>
            <div className="hero-stat">
              <span className="hero-stat-value">{formatCurrency(summary.avgRevenue)}</span>
              <span className="hero-stat-label">Avg Pred. Rev</span>
            </div>
            <div className="hero-stat">
              <span className="hero-stat-value">{formatPercent(summary.avgOcc)}</span>
              <span className="hero-stat-label">Avg Pred. Occ</span>
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
          min_price: 150000, max_price: 750000, bedrooms: null, property_type: null,
          sort_by: 'score', sort_order: 'desc', str_viable: true,
          mortgage_rate: null, mortgage_term: null, down_payment_pct: null,
        })}
      />

      {compareIds.size > 0 && (
        <div className="compare-bar">
          <span>{compareIds.size} of 3 selected</span>
          <button className="btn btn-primary" onClick={goCompare} disabled={compareIds.size < 2}>
            Compare
          </button>
          <button className="btn" onClick={() => setCompareIds(new Set())}>Clear</button>
        </div>
      )}

      {message && <div className="fallback-banner">{message}</div>}
      {error && <ErrorMessage message={error} onRetry={reload} />}
      {!error && (
        <div className={`property-results${loading ? ' results-loading' : ''}`}>
          {loading && <div className="results-overlay"><LoadingSpinner message="Updating..." /></div>}
          {properties.length > 0
            ? <PropertyList properties={properties} compareIds={compareIds} onCompareToggle={handleCompareToggle} />
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
