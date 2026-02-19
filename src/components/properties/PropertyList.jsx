import PropertyCard from './PropertyCard';

export default function PropertyList({ properties, variant = 'default', compareIds, onCompareToggle }) {
  if (!properties?.length) {
    return <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '40px' }}>No properties found.</p>;
  }

  const compact = variant === 'compact';

  return (
    <div className={`property-list${compact ? ' property-list--compact' : ''}`}>
      {properties.map(p => (
        <PropertyCard
          key={p.id}
          property={p}
          compact={compact}
          compareSelected={compareIds ? compareIds.has(p.id) : false}
          onCompareToggle={onCompareToggle}
        />
      ))}
    </div>
  );
}
