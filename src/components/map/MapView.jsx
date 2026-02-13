import { MapContainer, TileLayer } from 'react-leaflet';
import { AUSTIN_CENTER, DEFAULT_ZOOM } from '../../utils/constants';
import PropertyMarker from './PropertyMarker';

export default function MapView({ properties = [] }) {
  return (
    <MapContainer center={AUSTIN_CENTER} zoom={DEFAULT_ZOOM} scrollWheelZoom>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {properties.map(p => (
        <PropertyMarker key={p.id} property={p} />
      ))}
    </MapContainer>
  );
}
