export default function RangeSlider({ label, min, max, value, onChange, step = 1, format }) {
  return (
    <div className="filter-group">
      <label>{label}: {format ? format(value) : value}</label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ minWidth: 160 }}
      />
    </div>
  );
}
