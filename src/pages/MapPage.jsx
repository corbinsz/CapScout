import useProperties from '../hooks/useProperties';
import MapView from '../components/map/MapView';
import PropertyList from '../components/properties/PropertyList';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';

export default function MapPage() {
  const { properties, loading, error, reload } = useProperties({ per_page: 200 });

  if (loading) return <LoadingSpinner message="Loading map data..." />;
  if (error) return <ErrorMessage message={error} onRetry={reload} />;

  return (
    <div className="map-page">
      <div className="map-sidebar">
        <div className="map-sidebar-header">
          <h2>Investment Map</h2>
          <p>{properties.length} properties found</p>
        </div>
        <PropertyList properties={properties} variant="compact" />
      </div>
      <div className="map-container">
        <MapView properties={properties} />
      </div>
    </div>
  );
}
