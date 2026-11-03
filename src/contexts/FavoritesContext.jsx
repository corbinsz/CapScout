import { createContext, useContext, useState, useCallback } from 'react';

const STORAGE_KEY = 'capscout_favorites';

function loadFavorites() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
  } catch {
    return {};
  }
}

const FavoritesContext = createContext(null);

export function FavoritesProvider({ children }) {
  const [favorites, setFavorites] = useState(loadFavorites);

  const toggleFavorite = useCallback((property) => {
    setFavorites(prev => {
      const next = { ...prev };
      if (next[property.id]) {
        delete next[property.id];
      } else {
        // Store minimal data so Favorites page renders without API calls
        next[property.id] = {
          id: property.id,
          name: property.name,
          address: property.address,
          neighbourhood: property.neighbourhood,
          price: property.price,
          bedrooms: property.bedrooms,
          bathrooms: property.bathrooms,
          sqft: property.sqft,
          photo_url: property.photo_url,
          score: property.score,
          source: property.source,
          url: property.url,
          occupancy_rate: property.occupancy_rate,
          monthly_revenue: property.monthly_revenue,
          predicted: property.predicted,
          financials: property.financials,
          confidence_level: property.confidence_level,
          accommodates: property.accommodates,
          rating: property.rating,
        };
      }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const isFavorite = useCallback((id) => !!favorites[id], [favorites]);

  const favoritesList = Object.values(favorites);

  return (
    <FavoritesContext.Provider value={{ favorites, favoritesList, toggleFavorite, isFavorite }}>
      {children}
    </FavoritesContext.Provider>
  );
}

export function useFavorites() {
  const ctx = useContext(FavoritesContext);
  if (!ctx) throw new Error('useFavorites must be inside FavoritesProvider');
  return ctx;
}
