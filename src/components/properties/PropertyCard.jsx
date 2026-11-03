import { memo } from 'react';
import { Link } from 'react-router-dom';
import ScoreBadge from './ScoreBadge';
import FavoriteButton from './FavoriteButton';
import { formatCurrency, formatPercent } from '../../utils/formatters';
import { getScoreLabel } from '../../utils/scoreHelpers';
import { resolveBackendUrl } from '../../services/api';
import { useCity } from '../../contexts/CityContext';

export default memo(function PropertyCard({ property, compact = false, compareSelected = false, onCompareToggle }) {
  const { currentCity } = useCity();
  const {
    id, name, neighbourhood, address, photo_url, price, bedrooms, bathrooms,
    accommodates, occupancy_rate, score, monthly_revenue, predicted, financials, source, url,
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
    <Link to={`/property/${id}?city=${currentCity}`} className={`property-card${compact ? ' property-card--compact' : ''}`}>
      {!compact && (
        <div className="property-card-image">
          {photo_url ? (
            <>
              <img src={resolveBackendUrl(photo_url)} alt={photo_url.includes('openstreetmap') ? `Map preview near ${name || 'this home'}` : (name || 'Property')} loading="lazy" width={320} height={170} />
              {photo_url.includes('openstreetmap') && <span className="photo-caption">Map preview, not a photo</span>}
            </>
          ) : (
            <div className="property-card-placeholder">
              <span className="placeholder-icon">&#127968;</span>
              {locationText && <span className="placeholder-address">{locationText}</span>}
            </div>
          )}
          {isForSale && <ScoreBadge score={score} />}
          {isForSale && (property.confidence_level || property.predicted?.confidence?.level) && (
            <span className={`confidence-dot confidence-dot--${property.confidence_level || property.predicted?.confidence?.level}`} />
          )}
          <FavoriteButton property={property} />
          {onCompareToggle && (
            <button
              className={`compare-btn${compareSelected ? ' compare-btn--active' : ''}`}
              onClick={(e) => { e.preventDefault(); e.stopPropagation(); onCompareToggle(id); }}
              title={compareSelected ? 'Remove from comparison' : 'Add to comparison'}
            >
              {compareSelected ? '\u2713' : '+'}
            </button>
          )}
          {url && (
            <button className="external-link-btn" onClick={handleExternalClick} title="Open listing">
              Listing &#8599;
            </button>
          )}
        </div>
      )}

      <div className="property-card-body">
        <div className="property-card-price">
          {formatCurrency(price)}
          {isForSale ? null : <span> / night</span>}
          {isForSale && score != null && (
            <span className="score-word">{getScoreLabel(score)}</span>
          )}
        </div>
        {isForSale && property.insight && (
          <p className="insight-line">{property.insight}</p>
        )}
        {isForSale && property.explanation && (
          <p className="why-line">{property.explanation}</p>
        )}
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
                <span className="pcm-label">Pred. Nightly Rate</span>
                <span className="pcm-value">{formatCurrency(predicted?.predicted_nightly_rate)}</span>
              </div>
              <div className="pcm-item">
                <span className="pcm-label">Pred. Occupancy</span>
                <span className="pcm-value">{formatPercent(predicted?.predicted_occupancy_rate)}</span>
              </div>
              <div className="pcm-item">
                <span className="pcm-label">NOI / mo</span>
                <span className="pcm-value">{formatCurrency(financials?.noi != null ? financials.noi / 12 : null)}</span>
              </div>
              <div className="pcm-item">
                <span className="pcm-label">Cash Flow / mo</span>
                <span className={`pcm-value${(financials?.monthly_cashflow ?? 0) < 0 ? ' text-negative' : ''}`}>{formatCurrency(financials?.monthly_cashflow)}</span>
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
})
