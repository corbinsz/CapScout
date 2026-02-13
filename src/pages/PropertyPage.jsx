import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { MapContainer, TileLayer, CircleMarker } from 'react-leaflet';
import { fetchProperty } from '../services/propertyService';
import ScoreBadge from '../components/properties/ScoreBadge';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import { formatCurrency, formatPercent } from '../utils/formatters';
import { getScoreColor } from '../utils/scoreHelpers';

export default function PropertyPage() {
  const { id } = useParams();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    fetchProperty(id)
      .then(setData)
      .catch(err => setError(err.response?.data?.error || err.message))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return <LoadingSpinner message="Loading property..." />;
  if (error) return <ErrorMessage message={error} />;
  if (!data?.property) return <ErrorMessage message="Property not found" />;

  const { property, comparables, neighborhood_stats } = data;
  const fin = property.financials || {};
  const pred = property.predicted || {};
  const isForSale = property.source === 'rentcast' || property.source === 'realtor' || property.source === 'redfin';

  return (
    <div>
      {/* Breadcrumb */}
      <div className="breadcrumb">
        <Link to="/">Dashboard</Link>
        <span className="breadcrumb-sep">/</span>
        <span>{property.address || property.neighbourhood || 'Property'}</span>
        <span className="breadcrumb-sep">/</span>
        <span>{property.name || `#${property.id}`}</span>
      </div>

      {/* Hero Card */}
      <div className="property-hero">
        {/* Hero photo if available */}
        {property.photo_url && (
          <div className="property-hero-photo">
            <img src={property.photo_url} alt={property.address || property.name || 'Property'} />
          </div>
        )}

        <div className="property-hero-top">
          <div className="property-hero-info">
            <h1>{property.address || property.name || `Property #${property.id}`}</h1>
            <p className="hero-meta">
              {property.neighbourhood && `${property.neighbourhood} · `}
              {property.bedrooms} bd / {property.bathrooms} ba
              {!isForSale && property.accommodates ? ` · Sleeps ${property.accommodates}` : ''}
              {isForSale && property.sqft ? ` · ${property.sqft.toLocaleString()} sqft` : ''}
              {property.source && (
                <span className="hero-source-tag"> · {property.source === 'redfin' ? 'Redfin' : property.source === 'rentcast' ? 'RentCast' : property.source === 'realtor' ? 'Realtor.com' : 'Airbnb'}</span>
              )}
            </p>
          </div>
          <div className="property-hero-score">
            <ScoreBadge score={property.score} variant="bar" />
          </div>
        </div>

        {/* Key Metrics Strip */}
        <div className="key-metrics-strip">
          {isForSale ? (
            <>
              <div className="kms-item">
                <span className="kms-value">{formatCurrency(property.price)}</span>
                <span className="kms-label">Purchase Price</span>
              </div>
              <div className="kms-item">
                <span className="kms-value">{formatCurrency(pred.predicted_nightly_rate)}</span>
                <span className="kms-label">Pred. Nightly</span>
              </div>
              <div className="kms-item">
                <span className="kms-value">{formatCurrency(pred.predicted_monthly_revenue)}</span>
                <span className="kms-label">Pred. Monthly Rev</span>
              </div>
              <div className="kms-item">
                <span className="kms-value">{formatPercent(pred.predicted_occupancy_rate)}</span>
                <span className="kms-label">Pred. Occupancy</span>
              </div>
            </>
          ) : (
            <>
              <div className="kms-item">
                <span className="kms-value">{formatCurrency(property.price)}</span>
                <span className="kms-label">Nightly Rate</span>
              </div>
              <div className="kms-item">
                <span className="kms-value">{formatPercent(property.occupancy_rate)}</span>
                <span className="kms-label">Occupancy</span>
              </div>
              <div className="kms-item">
                <span className="kms-value">{formatCurrency(property.monthly_revenue)}</span>
                <span className="kms-label">Monthly Revenue</span>
              </div>
              <div className="kms-item">
                <span className="kms-value">{property.rating?.toFixed(2) ?? '—'}</span>
                <span className="kms-label">Rating</span>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="property-detail">
        {/* Predicted Performance */}
        {pred.predicted_nightly_rate && (
          <div className="detail-section">
            <div className="section-header">ML Predictions</div>
            <div className="section-body">
              <div className="detail-stats">
                <div className="detail-stat">
                  <span className="stat-label">Pred. Nightly</span>
                  <span className="stat-value">{formatCurrency(pred.predicted_nightly_rate)}</span>
                </div>
                <div className="detail-stat">
                  <span className="stat-label">Pred. Occupancy</span>
                  <span className="stat-value">{formatPercent(pred.predicted_occupancy_rate)}</span>
                </div>
                <div className="detail-stat">
                  <span className="stat-label">Pred. Monthly</span>
                  <span className="stat-value">{formatCurrency(pred.predicted_monthly_revenue)}</span>
                </div>
                <div className="detail-stat">
                  <span className="stat-label">Pred. Annual</span>
                  <span className="stat-value">{formatCurrency(pred.predicted_annual_revenue)}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Financial Analysis */}
        {fin.cap_rate != null && (
          <div className="detail-section">
            <div className="section-header">Financial Analysis</div>
            <div className="section-body">
              <div className="detail-stats">
                <div className="detail-stat">
                  <span className="stat-label">Cap Rate</span>
                  <span className="stat-value">{fin.cap_rate}%</span>
                </div>
                <div className="detail-stat">
                  <span className="stat-label">ROI</span>
                  <span className="stat-value">{fin.roi}%</span>
                </div>
                <div className="detail-stat">
                  <span className="stat-label">NOI</span>
                  <span className="stat-value">{formatCurrency(fin.noi)}</span>
                </div>
                <div className="detail-stat">
                  <span className="stat-label">Monthly Cashflow</span>
                  <span className="stat-value" style={{ color: fin.monthly_cashflow >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    {formatCurrency(fin.monthly_cashflow)}
                  </span>
                </div>
                <div className="detail-stat">
                  <span className="stat-label">Annual Revenue</span>
                  <span className="stat-value">{formatCurrency(fin.annual_revenue)}</span>
                </div>
                <div className="detail-stat">
                  <span className="stat-label">Annual Mortgage</span>
                  <span className="stat-value">{formatCurrency(fin.annual_mortgage)}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Mini Map */}
        {property.latitude && property.longitude && (
          <div className="detail-section">
            <div className="section-header">Location</div>
            <div className="section-body" style={{ padding: 0, height: 250 }}>
              <MapContainer
                center={[property.latitude, property.longitude]}
                zoom={14}
                scrollWheelZoom={false}
                style={{ height: '100%', width: '100%', borderRadius: '0 0 10px 10px' }}
              >
                <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                <CircleMarker
                  center={[property.latitude, property.longitude]}
                  radius={10}
                  pathOptions={{ fillColor: getScoreColor(property.score), color: '#fff', weight: 2, fillOpacity: 0.9 }}
                />
                {comparables?.map(c => (
                  <CircleMarker
                    key={c.id}
                    center={[c.latitude, c.longitude]}
                    radius={5}
                    pathOptions={{ fillColor: '#64748b', color: '#fff', weight: 1, fillOpacity: 0.6 }}
                  />
                ))}
              </MapContainer>
            </div>
          </div>
        )}

        {/* Neighborhood Stats */}
        {neighborhood_stats && (
          <div className="detail-section">
            <div className="section-header">Neighborhood: {neighborhood_stats.neighbourhood_cleansed}</div>
            <div className="section-body">
              <div className="detail-stats">
                <div className="detail-stat">
                  <span className="stat-label">Listings</span>
                  <span className="stat-value">{neighborhood_stats.count}</span>
                </div>
                <div className="detail-stat">
                  <span className="stat-label">Avg Price</span>
                  <span className="stat-value">{formatCurrency(neighborhood_stats.avg_price)}</span>
                </div>
                <div className="detail-stat">
                  <span className="stat-label">Avg Occupancy</span>
                  <span className="stat-value">{formatPercent(neighborhood_stats.avg_occupancy)}</span>
                </div>
                <div className="detail-stat">
                  <span className="stat-label">Avg Revenue</span>
                  <span className="stat-value">{formatCurrency(neighborhood_stats.avg_monthly_revenue)}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Comparable Listings */}
        {comparables?.length > 0 && (
          <div className="detail-section property-detail-full">
            <div className="section-header">Comparable STR Listings ({comparables.length})</div>
            <div className="section-body" style={{ overflowX: 'auto' }}>
              <table className="comps-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Beds</th>
                    <th>Nightly</th>
                    <th>Occupancy</th>
                    <th>Monthly Rev</th>
                    <th>Rating</th>
                    <th>Distance</th>
                  </tr>
                </thead>
                <tbody>
                  {comparables.map(c => (
                    <tr key={c.id}>
                      <td>{c.name?.slice(0, 40) || `#${c.id}`}</td>
                      <td>{c.bedrooms}</td>
                      <td>{formatCurrency(c.price)}</td>
                      <td>{formatPercent(c.occupancy_rate)}</td>
                      <td>{formatCurrency(c.monthly_revenue)}</td>
                      <td>{c.rating?.toFixed(1)}</td>
                      <td>{c.distance_km} km</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
