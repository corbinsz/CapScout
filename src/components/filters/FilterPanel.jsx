import { useState, useEffect, useRef } from 'react';
import { BEDROOM_OPTIONS, SORT_OPTIONS } from '../../utils/constants';

const SOURCE_OPTIONS = [
  { value: 'forsale', label: 'For Sale', desc: 'Homes for sale with STR projections' },
  { value: 'airbnb', label: 'Airbnb Data', desc: 'Real Airbnb listings powering our predictions' },
];

export default function FilterPanel({ filters, onFilterChange, onReset }) {
  const [open, setOpen] = useState(null);
  const panelRef = useRef(null);

  const toggle = (key) => setOpen(prev => prev === key ? null : key);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e) {
      if (panelRef.current && !panelRef.current.contains(e.target)) {
        setOpen(null);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  const handleChange = (key) => (e) => {
    const val = e.target.value;
    onFilterChange(key, val === '' ? null : isNaN(val) ? val : Number(val));
  };

  const currentSource = filters.source || 'forsale';
  const isForSale = currentSource !== 'airbnb';

  const hasActiveFilter = (key) => {
    if (key === 'price') return filters.min_price != null || filters.max_price != null;
    if (key === 'bedrooms') return filters.bedrooms != null;
    if (key === 'neighborhood') return !!filters.neighborhood;
    if (key === 'sort') return filters.sort_by && filters.sort_by !== 'score';
    if (key === 'financing') return filters.mortgage_rate != null || filters.mortgage_term != null || filters.down_payment_pct != null;
    return false;
  };

  const chipLabel = (key) => {
    if (key === 'price') {
      if (filters.min_price != null && filters.max_price != null)
        return isForSale
          ? `$${(filters.min_price / 1000).toFixed(0)}k–$${(filters.max_price / 1000).toFixed(0)}k`
          : `$${filters.min_price}–$${filters.max_price}`;
      if (filters.min_price != null)
        return isForSale ? `$${(filters.min_price / 1000).toFixed(0)}k+` : `$${filters.min_price}+`;
      if (filters.max_price != null)
        return isForSale ? `Up to $${(filters.max_price / 1000).toFixed(0)}k` : `Up to $${filters.max_price}`;
      return 'Price';
    }
    if (key === 'bedrooms') {
      const opt = BEDROOM_OPTIONS.find(o => String(o.value) === String(filters.bedrooms));
      return opt ? opt.label : 'Bedrooms';
    }
    if (key === 'neighborhood') return filters.neighborhood || 'Neighborhood';
    if (key === 'sort') {
      const opt = SORT_OPTIONS.find(o => o.value === filters.sort_by);
      return opt ? `Sort: ${opt.label}` : 'Sort';
    }
    if (key === 'financing') {
      const parts = [];
      if (filters.mortgage_rate != null) parts.push(`${filters.mortgage_rate}%`);
      if (filters.down_payment_pct != null) parts.push(`${filters.down_payment_pct}% down`);
      if (filters.mortgage_term != null) parts.push(`${filters.mortgage_term}yr`);
      return parts.length ? parts.join(' / ') : 'Financing';
    }
    return key;
  };

  return (
    <div className="filter-panel" ref={panelRef}>
      {/* Source toggle */}
      <div className="filter-chip source-toggle">
        {SOURCE_OPTIONS.map(opt => (
          <button
            key={opt.value}
            className={`filter-chip-btn${currentSource === opt.value ? ' active' : ''}`}
            title={opt.desc}
            onClick={() => onFilterChange('source', opt.value, { min_price: null, max_price: null })}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Price chip */}
      <div className="filter-chip">
        <button
          className={`filter-chip-btn${open === 'price' || hasActiveFilter('price') ? ' active' : ''}`}
          onClick={() => toggle('price')}
        >
          {chipLabel('price')}
          <span className="chip-arrow">&#9662;</span>
        </button>
        {open === 'price' && (
          <div className="filter-dropdown fd-price">
            <div className="fd-header">{isForSale ? 'Purchase Price Range' : 'Nightly Rate Range'}</div>
            <div className="fd-price-inputs">
              <div className="fd-price-field">
                <span className="fd-price-label">Min</span>
                <div className="fd-input-wrap">
                  <span className="fd-input-prefix">$</span>
                  <input
                    type="number"
                    placeholder={isForSale ? '100,000' : '0'}
                    value={filters.min_price ?? ''}
                    onChange={handleChange('min_price')}
                    autoFocus
                  />
                </div>
              </div>
              <span className="fd-price-divider" />
              <div className="fd-price-field">
                <span className="fd-price-label">Max</span>
                <div className="fd-input-wrap">
                  <span className="fd-input-prefix">$</span>
                  <input
                    type="number"
                    placeholder={isForSale ? '2,000,000' : '1,000'}
                    value={filters.max_price ?? ''}
                    onChange={handleChange('max_price')}
                  />
                </div>
              </div>
            </div>
            {hasActiveFilter('price') && (
              <button className="fd-clear" onClick={() => { onFilterChange('min_price', null); onFilterChange('max_price', null); }}>
                Clear price filter
              </button>
            )}
          </div>
        )}
      </div>

      {/* Bedrooms chip */}
      <div className="filter-chip">
        <button
          className={`filter-chip-btn${open === 'bedrooms' || hasActiveFilter('bedrooms') ? ' active' : ''}`}
          onClick={() => toggle('bedrooms')}
        >
          {chipLabel('bedrooms')}
          <span className="chip-arrow">&#9662;</span>
        </button>
        {open === 'bedrooms' && (
          <div className="filter-dropdown fd-bedrooms">
            <div className="fd-header">Bedrooms</div>
            <div className="fd-pill-group">
              {BEDROOM_OPTIONS.map(o => (
                <button
                  key={o.value}
                  className={`fd-pill${String(filters.bedrooms ?? '') === String(o.value) ? ' selected' : ''}`}
                  onClick={() => {
                    onFilterChange('bedrooms', o.value === '' ? null : Number(o.value));
                  }}
                >
                  {o.label}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Neighborhood chip — only for Airbnb */}
      {!isForSale && (
        <div className="filter-chip">
          <button
            className={`filter-chip-btn${open === 'neighborhood' || hasActiveFilter('neighborhood') ? ' active' : ''}`}
            onClick={() => toggle('neighborhood')}
          >
            {chipLabel('neighborhood')}
            <span className="chip-arrow">&#9662;</span>
          </button>
          {open === 'neighborhood' && (
            <div className="filter-dropdown fd-neighborhood">
              <div className="fd-header">Neighborhood</div>
              <div className="fd-input-wrap fd-search-wrap">
                <svg className="fd-search-icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="8.5" cy="8.5" r="5.5" />
                  <line x1="13" y1="13" x2="18" y2="18" />
                </svg>
                <input
                  type="text"
                  placeholder="Type a neighborhood name..."
                  value={filters.neighborhood ?? ''}
                  onChange={handleChange('neighborhood')}
                  autoFocus
                />
              </div>
              {filters.neighborhood && (
                <button className="fd-clear" onClick={() => onFilterChange('neighborhood', '')}>
                  Clear neighborhood
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {/* Sort chip */}
      <div className="filter-chip">
        <button
          className={`filter-chip-btn${open === 'sort' || hasActiveFilter('sort') ? ' active' : ''}`}
          onClick={() => toggle('sort')}
        >
          {chipLabel('sort')}
          <span className="chip-arrow">&#9662;</span>
        </button>
        {open === 'sort' && (
          <div className="filter-dropdown fd-sort">
            <div className="fd-header">Sort By</div>
            <div className="fd-option-list">
              {SORT_OPTIONS.map(o => (
                <button
                  key={o.value}
                  className={`fd-option${(filters.sort_by ?? 'score') === o.value ? ' selected' : ''}`}
                  onClick={() => {
                    onFilterChange('sort_by', o.value);
                    setOpen(null);
                  }}
                >
                  <span className="fd-option-radio" />
                  {o.label}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Financing chip — only for For Sale */}
      {isForSale && (
        <div className="filter-chip">
          <button
            className={`filter-chip-btn${open === 'financing' || hasActiveFilter('financing') ? ' active' : ''}`}
            onClick={() => toggle('financing')}
          >
            {chipLabel('financing')}
            <span className="chip-arrow">&#9662;</span>
          </button>
          {open === 'financing' && (
            <div className="filter-dropdown fd-financing">
              <div className="fd-header">Financing Assumptions</div>
              <div className="fd-financing-fields">
                <div className="fd-financing-field">
                  <span className="fd-price-label">Interest Rate %</span>
                  <div className="fd-input-wrap">
                    <input
                      type="number"
                      placeholder="7"
                      step="0.125"
                      min="0"
                      max="20"
                      value={filters.mortgage_rate ?? ''}
                      onChange={handleChange('mortgage_rate')}
                      autoFocus
                    />
                  </div>
                </div>
                <div className="fd-financing-field">
                  <span className="fd-price-label">Down Payment %</span>
                  <div className="fd-input-wrap">
                    <input
                      type="number"
                      placeholder="20"
                      step="5"
                      min="0"
                      max="100"
                      value={filters.down_payment_pct ?? ''}
                      onChange={handleChange('down_payment_pct')}
                    />
                  </div>
                </div>
                <div className="fd-financing-field">
                  <span className="fd-price-label">Loan Term (yr)</span>
                  <div className="fd-input-wrap">
                    <input
                      type="number"
                      placeholder="30"
                      step="5"
                      min="5"
                      max="40"
                      value={filters.mortgage_term ?? ''}
                      onChange={handleChange('mortgage_term')}
                    />
                  </div>
                </div>
              </div>
              {hasActiveFilter('financing') && (
                <button className="fd-clear" onClick={() => {
                  onFilterChange('mortgage_rate', null);
                  onFilterChange('mortgage_term', null);
                  onFilterChange('down_payment_pct', null);
                }}>
                  Reset to defaults
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {/* Confidence filter toggle — only for For Sale */}
      {isForSale && (
        <button
          className={`filter-chip-btn str-viable-btn${filters.str_viable ? ' active' : ''}`}
          onClick={() => onFilterChange('str_viable', !filters.str_viable)}
          title="Hide properties with low prediction confidence (not enough nearby Airbnb data)"
        >
          {filters.str_viable ? 'Confident Predictions Only' : 'All Confidence Levels'}
        </button>
      )}

      <button className="filter-reset-btn" onClick={onReset}>Reset</button>
    </div>
  );
}
