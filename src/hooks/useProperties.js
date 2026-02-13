import { useState, useEffect, useCallback } from 'react';
import { fetchProperties } from '../services/propertyService';

export default function useProperties(initialParams = {}) {
  const [properties, setProperties] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [message, setMessage] = useState(null);
  const [filters, setFilters] = useState({
    source: 'forsale',
    sort_by: 'score',
    sort_order: 'desc',
    per_page: 20,
    ...initialParams,
  });

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const data = await fetchProperties({ ...filters, page });
      setProperties(data.items || []);
      setTotal(data.total || 0);
      setMessage(data.message || null);
    } catch (err) {
      setError(err.response?.data?.error || err.message);
    } finally {
      setLoading(false);
    }
  }, [filters, page]);

  useEffect(() => { load(); }, [load]);

  const updateFilters = useCallback((newFilters) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
    setPage(1);
  }, []);

  return { properties, total, page, setPage, loading, error, message, filters, updateFilters, reload: load };
}
