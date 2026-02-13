import { formatCurrency, formatPercent } from '../../utils/formatters';

export default function PropertyPopup({ property }) {
  const { id, name, price, bedrooms, bathrooms, occupancy_rate, score, monthly_revenue } = property;
  return (
    <div>
      <strong>{name || `Property #${id}`}</strong>
      <div>{bedrooms} bd / {bathrooms} ba</div>
      <div>Nightly: {formatCurrency(price)}</div>
      <div>Occupancy: {formatPercent(occupancy_rate)}</div>
      <div>Monthly Rev: {formatCurrency(monthly_revenue)}</div>
      {score != null && <div>Score: <b>{score.toFixed(1)}</b></div>}
      <div style={{ marginTop: 4 }}>
        <a href={`/property/${id}`}>View Details</a>
      </div>
    </div>
  );
}
