import { useMemo, useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import useProperties from '../hooks/useProperties';
import { useCity } from '../contexts/CityContext';
import { useAuth } from '../contexts/AuthContext';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import FilterPanel from '../components/filters/FilterPanel';
import PropertyList from '../components/properties/PropertyList';
import { formatCurrency, formatPercent, formatNumber } from '../utils/formatters';
import api from '../services/api';

export default function Dashboard() {
  const { properties, total, totalAvailable, page, setPage, loading, error, filters, updateFilters, reload, message } = useProperties();
  const { cityMeta, currentCity } = useCity();
  const { user, isPro, openAuth, openPaywall } = useAuth();
  const navigate = useNavigate();
  const [metrics, setMetrics] = useState(null);
  const [reportError, setReportError] = useState('');

  useEffect(() => {
    api.get('/model-metrics', { params: { city: currentCity } })
      .then(res => setMetrics(res.data?.available ? res.data : null))
      .catch(() => setMetrics(null));
  }, [currentCity]);

  const downloadReport = async () => {
    setReportError('');
    if (!user) {
      openAuth();
      return;
    }
    if (!isPro) {
      openPaywall();
      return;
    }
    try {
      const res = await api.get('/reports/top-deals', {
        params: { city: currentCity },
        responseType: 'blob',
      });
      const blob = new Blob([res.data], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `capscout-top-deals-${currentCity}.csv`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      const status = err.response?.status;
      if (status === 401) openAuth();
      else if (status === 402) openPaywall();
      else setReportError('The report could not be downloaded.');
    }
  };
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
        <p className="eyebrow">Short-term rental screener · {cityDisplay}</p>
        <h1>Which homes can the rent actually carry?</h1>
        <p className="hero-detail">
          CapScout estimates a nightly rate and occupancy, then scores how that income covers a
          mortgage, taxes, insurance, and operating costs. 70+ is excellent, 55 is good, 45 is fair,
          and anything lower is weak under the loan assumptions in the filters.
        </p>
        <div className="legend-row" aria-label="Score legend">
          <span className="legend-chip legend-excellent">70+ Excellent</span>
          <span className="legend-chip legend-good">55 Good</span>
          <span className="legend-chip legend-fair">45 Fair</span>
          <span className="legend-chip legend-weak">Below 45 Weak</span>
        </div>
        {message && <p className="source-note">{message}</p>}
        {metrics?.holdout?.price_raw?.mae != null && (
          <p className="source-note">
            On homes held out of training, a typical nightly-rate miss in {cityDisplay} is about {formatCurrency(metrics.holdout.price_raw.mae)}.
            Occupancy is harder: the calendar cannot tell a booking from a host block.
          </p>
        )}
        <div className="hero-actions">
          <button className="btn btn-primary" type="button" onClick={downloadReport}>
            Top deals in {cityMeta?.name || 'this city'}
          </button>
          <span className="hero-action-note">{isPro ? 'Pro download' : 'Free to browse · Pro report'}</span>
        </div>
        {reportError && <p className="form-error">{reportError}</p>}
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
