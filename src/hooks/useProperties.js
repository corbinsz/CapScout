import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchProperties } from '../services/propertyService';
import { useCity } from '../contexts/CityContext';

const DEBOUNCE_MS = 350;

export default function useProperties(initialParams = {}) {
  const { currentCity } = useCity();
  const [properties, setProperties] = useState([]);
  const [total, setTotal] = useState(0);
  const [totalAvailable, setTotalAvailable] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [message, setMessage] = useState(null);
  const [filters, setFilters] = useState({
    source: 'forsale',
    sort_by: 'score',
    sort_order: 'desc',
    per_page: 20,
    str_viable: true,
    min_price: 150000,
    max_price: 750000,
    bedrooms: null,
    ...initialParams,
  });

  const debounceRef = useRef(null);
  const abortRef = useRef(null);

  const load = useCallback(async (immediate = false) => {
    // Cancel any pending debounce
    if (debounceRef.current) clearTimeout(debounceRef.current);
    // Abort any in-flight request
    if (abortRef.current) abortRef.current.abort();

    const doFetch = async () => {
      const controller = new AbortController();
      abortRef.current = controller;
      setLoading(true);
      setError(null);
      try {
        const data = await fetchProperties(
          { ...filters, page, city: currentCity },
          { signal: controller.signal }
        );
        if (!controller.signal.aborted) {
          setProperties(data.items || []);
          setTotal(data.total || 0);
          setTotalAvailable(data.total_available || data.total || 0);
          setMessage(data.message || null);
        }
      } catch (err) {
        if (err.name === 'CanceledError' || err.name === 'AbortError') return;
        const msg = err.response?.data?.error || err.message;
        setError(typeof msg === 'string' ? msg : 'Failed to load properties');
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    };

    if (immediate) {
      doFetch();
    } else {
      setLoading(true);
      debounceRef.current = setTimeout(doFetch, DEBOUNCE_MS);
    }
  }, [filters, page, currentCity]);

  useEffect(() => { load(); }, [load]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      if (abortRef.current) abortRef.current.abort();
    };
  }, []);

  // Reset to page 1 when city changes
  useEffect(() => { setPage(1); }, [currentCity]);

  const updateFilters = useCallback((newFilters) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
    setPage(1);
  }, []);

  return { properties, total, totalAvailable, page, setPage, loading, error, message, filters, updateFilters, reload: () => load(true) };
}
