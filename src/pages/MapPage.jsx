import useProperties from '../hooks/useProperties';
import MapView from '../components/map/MapView';
import PropertyList from '../components/properties/PropertyList';
import FilterPanel from '../components/filters/FilterPanel';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';

export default function MapPage() {
  const { properties, loading, error, reload, filters, updateFilters } = useProperties({ per_page: 200 });

  return (
    <div className="map-page">
      <div className="map-sidebar">
        <div className="map-sidebar-header">
          <h2>Investment Map</h2>
          <p>{properties.length} properties found</p>
        </div>
        <div className="map-sidebar-filters">
          <FilterPanel
            filters={filters}
            onFilterChange={(key, val, extra) => updateFilters({ [key]: val, ...extra })}
            onReset={() => updateFilters({
              min_price: 150000, max_price: 750000, bedrooms: null, property_type: null,
              sort_by: 'score', sort_order: 'desc', str_viable: true,
              mortgage_rate: null, mortgage_term: null, down_payment_pct: null,
            })}
          />
        </div>
        {loading && <LoadingSpinner message="Loading..." />}
        {error && <ErrorMessage message={error} onRetry={reload} />}
        {!loading && !error && <PropertyList properties={properties} variant="compact" />}
      </div>
      <div className="map-container">
        <MapView properties={properties} />
      </div>
    </div>
  );
}
