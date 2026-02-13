import { useState } from 'react';
import { BEDROOM_OPTIONS, SORT_OPTIONS } from '../../utils/constants';

const SOURCE_OPTIONS = [
  { value: 'forsale', label: 'For Sale' },
  { value: 'airbnb', label: 'Airbnb Rentals' },
];

export default function FilterPanel({ filters, onFilterChange, onReset }) {
  const [open, setOpen] = useState(null);

  const toggle = (key) => setOpen(prev => prev === key ? null : key);

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
    return key;
  };

  return (
    <div className="filter-panel">
      {/* Source toggle */}
      <div className="filter-chip source-toggle">
        {SOURCE_OPTIONS.map(opt => (
          <button
            key={opt.value}
            className={`filter-chip-btn${currentSource === opt.value ? ' active' : ''}`}
            onClick={() => {
              onFilterChange('source', opt.value);
              // Reset price filters when switching source (different ranges)
              onFilterChange('min_price', null);
              onFilterChange('max_price', null);
            }}
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
          <div className="filter-dropdown">
            <label>{isForSale ? 'Purchase price range' : 'Nightly rate range'}</label>
            <div className="filter-dropdown-row">
              <input
                type="number"
                placeholder={isForSale ? '100000' : '0'}
                value={filters.min_price ?? ''}
                onChange={handleChange('min_price')}
              />
              <span>–</span>
              <input
                type="number"
                placeholder={isForSale ? '2000000' : '1000'}
                value={filters.max_price ?? ''}
                onChange={handleChange('max_price')}
              />
            </div>
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
          <div className="filter-dropdown">
            <label>Bedrooms</label>
            <select value={filters.bedrooms ?? ''} onChange={handleChange('bedrooms')}>
              {BEDROOM_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
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
            <div className="filter-dropdown">
              <label>Neighborhood</label>
              <input
                type="text"
                placeholder="Search neighborhoods..."
                value={filters.neighborhood ?? ''}
                onChange={handleChange('neighborhood')}
                autoFocus
              />
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
          <div className="filter-dropdown">
            <label>Sort by</label>
            <select value={filters.sort_by ?? 'score'} onChange={handleChange('sort_by')}>
              {SORT_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
        )}
      </div>

      <button className="filter-reset-btn" onClick={onReset}>Reset</button>
    </div>
  );
}
