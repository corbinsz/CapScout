export function formatCurrency(value) {
  if (value == null) return '—';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

export function formatPercent(value, decimals = 1) {
  if (value == null) return '—';
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatNumber(value, decimals = 0) {
  if (value == null) return '—';
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

export function formatScore(score) {
  if (score == null) return '—';
  return score.toFixed(1);
}
