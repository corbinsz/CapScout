import { CircleMarker, Popup } from 'react-leaflet';
import { getScoreColor } from '../../utils/scoreHelpers';
import PropertyPopup from './PropertyPopup';

export default function PropertyMarker({ property }) {
  const { latitude, longitude, score } = property;
  if (!latitude || !longitude) return null;

  return (
    <CircleMarker
      center={[latitude, longitude]}
      radius={7}
      pathOptions={{
        fillColor: getScoreColor(score),
        color: '#fff',
        weight: 2,
        fillOpacity: 0.85,
      }}
    >
      <Popup>
        <PropertyPopup property={property} />
      </Popup>
    </CircleMarker>
  );
}
