import { useState, useCallback } from 'react';
import Map, { Marker, Popup, NavigationControl } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { AUSTIN_CENTER, DEFAULT_ZOOM } from '../../utils/constants';
import { useCity } from '../../contexts/CityContext';
import { getScoreColor } from '../../utils/scoreHelpers';
import PropertyPopup from './PropertyPopup';

const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json';

export default function MapView({ properties = [] }) {
  const { currentCity, cityMeta } = useCity();
  const lat = cityMeta?.latitude ?? AUSTIN_CENTER[0];
  const lng = cityMeta?.longitude ?? AUSTIN_CENTER[1];
  const zoom = cityMeta?.default_zoom || DEFAULT_ZOOM;
  const [popup, setPopup] = useState(null);

  const handleMarkerClick = useCallback((p) => {
    setPopup(p);
  }, []);

  return (
    <Map
      key={currentCity}
      initialViewState={{ longitude: lng, latitude: lat, zoom }}
      style={{ width: '100%', height: '100%' }}
      mapStyle={MAP_STYLE}
    >
      <NavigationControl position="top-right" />
      {properties.map(p => {
        if (!p.latitude || !p.longitude) return null;
        return (
          <Marker
            key={p.id}
            longitude={p.longitude}
            latitude={p.latitude}
            anchor="center"
            onClick={(e) => { e.originalEvent.stopPropagation(); handleMarkerClick(p); }}
          >
            <div
              className="map-dot"
              style={{ background: getScoreColor(p.score) }}
            />
          </Marker>
        );
      })}
      {popup && (
        <Popup
          longitude={popup.longitude}
          latitude={popup.latitude}
          anchor="bottom"
          onClose={() => setPopup(null)}
          closeOnClick={false}
          className="map-popup"
        >
          <PropertyPopup property={popup} />
        </Popup>
      )}
    </Map>
  );
}
