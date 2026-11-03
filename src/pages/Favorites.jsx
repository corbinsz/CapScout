import { Link } from 'react-router-dom';
import { useFavorites } from '../contexts/FavoritesContext';
import PropertyList from '../components/properties/PropertyList';

export default function Favorites() {
  const { favoritesList } = useFavorites();

  return (
    <div>
      <div className="hero-section">
        <h1>Saved homes</h1>
        <p>{favoritesList.length} {favoritesList.length === 1 ? 'home' : 'homes'} saved on this device</p>
      </div>

      {favoritesList.length > 0 ? (
        <PropertyList properties={favoritesList} />
      ) : (
        <div className="empty-state">
          <span className="empty-state-icon">{'\u2661'}</span>
          <h3>Nothing saved yet</h3>
          <p>Use the heart on a deal to keep it here. Saved homes stay in this browser.</p>
          <p><Link to="/deals">Browse deals</Link></p>
        </div>
      )}
    </div>
  );
}
