import { useCity } from '../../contexts/CityContext';

export default function CitySelector() {
  const { cities, currentCity, switchCity } = useCity();

  const available = cities.filter(c => c.has_data);
  if (available.length === 0) return null;

  return (
    <select
      className="city-selector"
      value={currentCity}
      onChange={e => switchCity(e.target.value)}
    >
      {available.map(c => (
        <option key={c.key} value={c.key}>
          {c.display}{!c.has_models ? ' (no models)' : ''}
        </option>
      ))}
    </select>
  );
}
