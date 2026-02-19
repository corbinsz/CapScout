import { useFavorites } from '../../contexts/FavoritesContext';

export default function FavoriteButton({ property }) {
  const { isFavorite, toggleFavorite } = useFavorites();
  const active = isFavorite(property.id);

  const handleClick = (e) => {
    e.preventDefault();
    e.stopPropagation();
    toggleFavorite(property);
  };

  return (
    <button
      className={`favorite-btn${active ? ' favorite-btn--active' : ''}`}
      onClick={handleClick}
      title={active ? 'Remove from favorites' : 'Save to favorites'}
      aria-label={active ? 'Remove from favorites' : 'Save to favorites'}
    >
      {active ? '\u2665' : '\u2661'}
    </button>
  );
}
