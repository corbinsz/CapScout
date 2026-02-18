import { Link } from 'react-router-dom';
import ScoreBadge from './ScoreBadge';
import { formatCurrency, formatPercent } from '../../utils/formatters';
import { resolveBackendUrl } from '../../services/api';

export default function PropertyCard({ property, compact = false }) {
  const {
    id, name, neighbourhood, address, photo_url, price, bedrooms, bathrooms,
    accommodates, occupancy_rate, score, monthly_revenue, predicted, source, url,
  } = property;

  const isForSale = source === 'rentcast' || source === 'realtor' || source === 'redfin';
  const locationText = address || neighbourhood || '';

  const handleExternalClick = (e) => {
    if (url) {
      e.preventDefault();
      e.stopPropagation();
      window.open(url, '_blank', 'noopener');
    }
  };

  return (
    <Link to={`/property/${id}`} className={`property-card${compact ? ' property-card--compact' : ''}`}>
      {!compact && (
        <div className="property-card-image">
          {photo_url ? (
            <img src={resolveBackendUrl(photo_url)} alt={name || 'Property'} />
          ) : (
            <div className="property-card-placeholder">
              <span className="placeholder-icon">&#127968;</span>
              {locationText && <span className="placeholder-address">{locationText}</span>}
            </div>
          )}
          {isForSale && <ScoreBadge score={score} />}
          {url && (
            <button className="external-link-btn" onClick={handleExternalClick} title="View on Redfin">
              Redfin &#8599;
            </button>
          )}
        </div>
      )}

      <div className="property-card-body">
        <div className="property-card-price">
          {formatCurrency(price)}
          {isForSale ? null : <span> / night</span>}
        </div>
        {locationText && (
          <div className="property-card-address">{locationText}</div>
        )}
        <div className="property-card-specs">
          <span>{bedrooms} bd</span>
          <span className="spec-dot" />
          <span>{bathrooms} ba</span>
          {!isForSale && accommodates > 0 && (
            <>
              <span className="spec-dot" />
              <span>Sleeps {accommodates}</span>
            </>
          )}
          {property.sqft > 0 && isForSale && (
            <>
              <span className="spec-dot" />
              <span>{property.sqft.toLocaleString()} sqft</span>
            </>
          )}
        </div>
        {!locationText && name && (
          <div className="property-card-name">{name || `Property #${id}`}</div>
        )}
      </div>

      {!compact && (
        <div className="property-card-metrics">
          {isForSale ? (
            <>
              <div className="pcm-item">
                <span className="pcm-label">Pred. Nightly</span>
                <span className="pcm-value">{formatCurrency(predicted?.predicted_nightly_rate)}</span>
              </div>
              <div className="pcm-item">
                <span className="pcm-label">Pred. Revenue</span>
                <span className="pcm-value">{formatCurrency(predicted?.predicted_monthly_revenue)}</span>
              </div>
              <div className="pcm-item">
                <span className="pcm-label">STR Area</span>
                <span className={`pcm-value str-tag str-${property.str_activity || 'unknown'}`}>
                  {property.str_activity === 'active' ? 'Active' : property.str_activity === 'low' ? 'Low' : property.str_activity === 'none' ? 'None' : '—'}
                </span>
              </div>
            </>
          ) : (
            <>
              <div className="pcm-item">
                <span className="pcm-label">Occupancy</span>
                <span className="pcm-value">{formatPercent(occupancy_rate)}</span>
              </div>
              <div className="pcm-item">
                <span className="pcm-label">Monthly Rev</span>
                <span className="pcm-value">{formatCurrency(monthly_revenue)}</span>
              </div>
              <div className="pcm-item">
                <span className="pcm-label">Rating</span>
                <span className="pcm-value">{property.rating?.toFixed(1) ?? '—'}</span>
              </div>
            </>
          )}
        </div>
      )}
    </Link>
  );
}
