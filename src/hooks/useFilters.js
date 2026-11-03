import { useState, useCallback } from 'react';

export default function useFilters(defaults = {}) {
  const [filters, setFilters] = useState({
    min_price: null,
    max_price: null,
    bedrooms: null,
    neighborhood: '',
    min_score: null,
    sort_by: 'score',
    sort_order: 'desc',
    ...defaults,
  });

  const updateFilter = useCallback((key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  }, []);

  const resetFilters = useCallback(() => {
    setFilters({
      min_price: null,
      max_price: null,
      bedrooms: null,
      neighborhood: '',
      min_score: null,
      sort_by: 'score',
      sort_order: 'desc',
      ...defaults,
    });
  }, [defaults]);

  // Build query params (strip nulls/empty)
  const queryParams = Object.fromEntries(
    Object.entries(filters).filter(([, v]) => v != null && v !== '')
  );

  return { filters, updateFilter, resetFilters, queryParams };
}
