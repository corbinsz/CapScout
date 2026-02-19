import { useFavorites } from '../contexts/FavoritesContext';
import PropertyList from '../components/properties/PropertyList';

export default function Favorites() {
  const { favoritesList } = useFavorites();

  return (
    <div>
      <div className="hero-section">
        <h1>Saved Favorites</h1>
        <p>{favoritesList.length} {favoritesList.length === 1 ? 'property' : 'properties'} saved</p>
      </div>

      {favoritesList.length > 0 ? (
        <PropertyList properties={favoritesList} />
      ) : (
        <div className="empty-state">
          <span className="empty-state-icon">{'\u2661'}</span>
          <h3>No favorites yet</h3>
          <p>Click the heart icon on any property card to save it here for quick access.</p>
        </div>
      )}
    </div>
  );
}
