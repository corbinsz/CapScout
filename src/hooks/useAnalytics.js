import { useState, useEffect } from 'react';
import { fetchAnalytics } from '../services/analyticsService';
import { useCity } from '../contexts/CityContext';

export default function useAnalytics() {
  const { currentCity } = useCity();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setData(null);
    fetchAnalytics(currentCity)
      .then(result => { if (!cancelled) setData(result); })
      .catch(err => { if (!cancelled) setError(err.response?.data?.error || err.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [currentCity]);

  return { data, loading, error };
}
