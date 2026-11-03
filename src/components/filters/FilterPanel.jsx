import { useState, useEffect, useRef } from 'react';
import { BEDROOM_OPTIONS, PROPERTY_TYPE_OPTIONS, SORT_OPTIONS } from '../../utils/constants';
import CitySelector from '../layout/CitySelector';

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

  const hasActiveFilter = (key) => {
    if (key === 'price') return filters.min_price != null || filters.max_price != null;
    if (key === 'bedrooms') return filters.bedrooms != null;
    if (key === 'sort') return filters.sort_by && filters.sort_by !== 'score';
    if (key === 'property_type') return filters.property_type != null;
    if (key === 'financing') return filters.mortgage_rate != null || filters.mortgage_term != null || filters.down_payment_pct != null;
    return false;
  };

  const chipLabel = (key) => {
    if (key === 'price') {
      if (filters.min_price != null && filters.max_price != null)
        return `$${(filters.min_price / 1000).toFixed(0)}k\u2013$${(filters.max_price / 1000).toFixed(0)}k`;
      if (filters.min_price != null)
        return `$${(filters.min_price / 1000).toFixed(0)}k+`;
      if (filters.max_price != null)
        return `Up to $${(filters.max_price / 1000).toFixed(0)}k`;
      return 'Price';
    }
    if (key === 'bedrooms') {
      const opt = BEDROOM_OPTIONS.find(o => String(o.value) === String(filters.bedrooms));
      return opt ? opt.label : 'Bedrooms';
    }
    if (key === 'sort') {
      const opt = SORT_OPTIONS.find(o => o.value === filters.sort_by);
      return opt ? `Sort: ${opt.label}` : 'Sort';
    }
    if (key === 'property_type') {
      const opt = PROPERTY_TYPE_OPTIONS.find(o => o.value === filters.property_type);
      return opt ? opt.label : 'Type';
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
      {/* City selector */}
      <CitySelector />

      {/* Price chip */}
      <div className="filter-chip">
        <button
          className={`filter-chip-btn${open === 'price' || hasActiveFilter('price') ? ' active' : ''}`}
          type="button"
          aria-expanded={open === 'price'}
          onClick={() => toggle('price')}
        >
          {chipLabel('price')}
          <span className="chip-arrow">&#9662;</span>
        </button>
        {open === 'price' && (
          <div className="filter-dropdown fd-price">
            <div className="fd-header">Purchase Price Range</div>
            <div className="fd-price-inputs">
              <div className="fd-price-field">
                <span className="fd-price-label">Min</span>
                <div className="fd-input-wrap">
                  <span className="fd-input-prefix">$</span>
                  <input
                    type="number"
                    placeholder="100,000"
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
                    placeholder="2,000,000"
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
          type="button"
          aria-expanded={open === 'bedrooms'}
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

      {/* Property Type chip */}
      <div className="filter-chip">
        <button
          className={`filter-chip-btn${open === 'property_type' || hasActiveFilter('property_type') ? ' active' : ''}`}
          type="button"
          aria-expanded={open === 'property_type'}
          onClick={() => toggle('property_type')}
        >
          {chipLabel('property_type')}
          <span className="chip-arrow">&#9662;</span>
        </button>
        {open === 'property_type' && (
          <div className="filter-dropdown fd-bedrooms">
            <div className="fd-header">Property Type</div>
            <div className="fd-pill-group">
              {PROPERTY_TYPE_OPTIONS.map(o => (
                <button
                  key={o.value}
                  className={`fd-pill${(filters.property_type ?? '') === o.value ? ' selected' : ''}`}
                  onClick={() => {
                    onFilterChange('property_type', o.value === '' ? null : o.value);
                  }}
                >
                  {o.label}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Sort chip */}
      <div className="filter-chip">
        <button
          className={`filter-chip-btn${open === 'sort' || hasActiveFilter('sort') ? ' active' : ''}`}
          type="button"
          aria-expanded={open === 'sort'}
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

      {/* Financing chip */}
      <div className="filter-chip">
          <button
            className={`filter-chip-btn${open === 'financing' || hasActiveFilter('financing') ? ' active' : ''}`}
            type="button"
            aria-expanded={open === 'financing'}
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

      {/* Confidence filter toggle */}
      <button
        className={`filter-chip-btn str-viable-btn${filters.str_viable ? ' active' : ''}`}
        onClick={() => onFilterChange('str_viable', !filters.str_viable)}
        title="Hide properties with low prediction confidence (not enough nearby Airbnb data)"
      >
        {filters.str_viable ? 'Confident Predictions Only' : 'All Confidence Levels'}
      </button>

      <button className="filter-reset-btn" onClick={onReset}>Reset</button>
    </div>
  );
}
