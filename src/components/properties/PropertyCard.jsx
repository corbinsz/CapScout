import { Link } from 'react-router-dom';
import ScoreBadge from './ScoreBadge';
import { formatCurrency, formatPercent } from '../../utils/formatters';

export default function PropertyCard({ property, compact = false }) {
  const {
    id, name, neighbourhood, address, photo_url, price, bedrooms, bathrooms,
    accommodates, occupancy_rate, score, monthly_revenue, predicted, source,
  } = property;

  const isForSale = source === 'rentcast' || source === 'realtor' || source === 'redfin';
  const locationText = address || neighbourhood || '';

  return (
    <Link to={`/property/${id}`} className={`property-card${compact ? ' property-card--compact' : ''}`}>
      {!compact && (
        <div className="property-card-image">
          {photo_url ? (
            <img src={photo_url} alt={name || 'Property'} />
          ) : (
            <div className="property-card-placeholder">
              <span className="placeholder-icon">&#127968;</span>
              {locationText && <span className="placeholder-address">{locationText}</span>}
            </div>
          )}
          <ScoreBadge score={score} />
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
                <span className="pcm-label">Score</span>
                <span className="pcm-value">{score?.toFixed(1) ?? '—'}</span>
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
                <span className="pcm-label">Score</span>
                <span className="pcm-value">{score?.toFixed(1) ?? '—'}</span>
              </div>
            </>
          )}
        </div>
      )}
    </Link>
  );
}
