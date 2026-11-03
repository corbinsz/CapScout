import { useState, useEffect } from 'react';
import { fetchAnalytics } from '../services/analyticsService';
import { useCity } from '../contexts/CityContext';

export default function useAnalytics() {
  const { currentCity } = useCity();
  const [loaded, setLoaded] = useState(null);

  useEffect(() => {
    let cancelled = false;
    fetchAnalytics(currentCity)
      .then(result => { if (!cancelled) setLoaded({ city: currentCity, data: result, error: null }); })
      .catch(err => {
        if (!cancelled) setLoaded({ city: currentCity, data: null, error: err.response?.data?.error || err.message });
      });
    return () => { cancelled = true; };
  }, [currentCity]);

  const ready = loaded && loaded.city === currentCity;
  return {
    data: ready ? loaded.data : null,
    loading: !ready,
    error: ready ? loaded.error : null,
  };
}
