import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import api from '../services/api';

const CityContext = createContext(null);

const DEFAULT_CITY = 'austin';

export function CityProvider({ children }) {
  const [cities, setCities] = useState([]);
  const [currentCity, setCurrentCity] = useState(
    () => localStorage.getItem('capscout_city') || DEFAULT_CITY
  );

  useEffect(() => {
    api.get('/cities')
      .then(res => setCities(res.data))
      .catch(() => {});
  }, []);

  const switchCity = useCallback((key) => {
    setCurrentCity(key);
    localStorage.setItem('capscout_city', key);
  }, []);

  const cityMeta = cities.find(c => c.key === currentCity) || null;

  return (
    <CityContext.Provider value={{ cities, currentCity, cityMeta, switchCity }}>
      {children}
    </CityContext.Provider>
  );
}

export function useCity() {
  const ctx = useContext(CityContext);
  if (!ctx) throw new Error('useCity must be used within CityProvider');
  return ctx;
}
