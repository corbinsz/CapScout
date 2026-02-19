import { useCity } from '../../contexts/CityContext';

const CITY_LABELS = {
  austin: 'Austin, TX',
  nashville: 'Nashville, TN',
  'san-diego': 'San Diego, CA',
  dallas: 'Dallas, TX',
  asheville: 'Asheville, NC',
  seattle: 'Seattle, WA',
};

export default function CitySelector() {
  const { cities, currentCity, switchCity } = useCity();

  const available = cities.filter(c => c.has_data);

  // Fallback: if API hasn't loaded yet, show at least the current city
  if (available.length === 0) {
    return (
      <select className="city-selector" value={currentCity} disabled>
        <option value={currentCity}>{CITY_LABELS[currentCity] || currentCity}</option>
      </select>
    );
  }

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
