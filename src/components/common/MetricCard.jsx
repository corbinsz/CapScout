export default function MetricCard({ label, value, subtitle, color, accentColor }) {
  return (
    <div className="metric-card" style={accentColor ? { '--accent': accentColor } : undefined}>
      {accentColor && (
        <style>{`.metric-card[style*="--accent"]::before { background: var(--accent); }`}</style>
      )}
      <span className="metric-label">{label}</span>
      <span className="metric-value" style={color ? { color } : undefined}>{value}</span>
      {subtitle && <span className="metric-subtitle">{subtitle}</span>}
    </div>
  );
}
