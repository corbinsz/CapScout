import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useCity } from '../../contexts/CityContext';
import { searchProperties } from '../../services/propertyService';

export default function SearchBar() {
  const { currentCity } = useCity();
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [activeIdx, setActiveIdx] = useState(-1);
  const wrapperRef = useRef(null);
  const debounceRef = useRef(null);

  const doSearch = useCallback(async (q) => {
    if (q.length < 2) {
      setResults([]);
      setOpen(false);
      return;
    }
    setLoading(true);
    try {
      const res = await searchProperties(q, currentCity);
      setResults(res);
      setOpen(res.length > 0);
      setActiveIdx(-1);
    } catch {
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [currentCity]);

  useEffect(() => {
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => doSearch(query), 300);
    return () => clearTimeout(debounceRef.current);
  }, [query, doSearch]);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  function handleSelect(item) {
    setOpen(false);
    setQuery('');
    setResults([]);
    navigate(`/property/${item.id}?city=${currentCity}`);
  }

  function handleKeyDown(e) {
    if (!open || results.length === 0) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIdx(prev => Math.min(prev + 1, results.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIdx(prev => Math.max(prev - 1, 0));
    } else if (e.key === 'Enter' && activeIdx >= 0) {
      e.preventDefault();
      handleSelect(results[activeIdx]);
    } else if (e.key === 'Escape') {
      setOpen(false);
    }
  }

  const fmtPrice = (p) => p >= 1000 ? `$${(p / 1000).toFixed(0)}k` : `$${p}`;

  return (
    <div className="search-bar-wrapper" ref={wrapperRef}>
      <div className="search-bar-input-wrap">
        <svg className="search-bar-icon" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="8.5" cy="8.5" r="5.5" />
          <line x1="13" y1="13" x2="18" y2="18" />
        </svg>
        <input
          type="text"
          className="search-bar-input"
          placeholder="Search properties by address..."
          value={query}
          onChange={e => setQuery(e.target.value)}
          onFocus={() => results.length > 0 && setOpen(true)}
          onKeyDown={handleKeyDown}
        />
        {loading && <span className="search-bar-spinner" />}
      </div>

      {open && (
        <div className="search-dropdown">
          {results.map((item, i) => (
            <button
              key={`${item.source}-${item.id}`}
              className={`search-result-item${i === activeIdx ? ' active' : ''}`}
              onMouseEnter={() => setActiveIdx(i)}
              onClick={() => handleSelect(item)}
            >
              <div className="search-result-main">
                <span className="search-result-name">{item.name || item.address}</span>
                <span className="search-result-meta">
                  {item.bedrooms}bd
                  {item.bathrooms ? ` / ${item.bathrooms}ba` : ''}
                  {item.price ? ` · ${fmtPrice(item.price)}` : ''}
                </span>
              </div>
              <span className={`search-result-source ${item.source}`}>
                {item.source === 'redfin' ? 'For Sale' : 'Airbnb'}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
