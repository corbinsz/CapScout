import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { formatCurrency, formatPercent } from '../../utils/formatters';

export default function RevenueChart({ neighborhoods = [] }) {
  const data = neighborhoods.slice(0, 10).map(n => ({
    name: n.neighbourhood_cleansed?.length > 15
      ? n.neighbourhood_cleansed.slice(0, 13) + '...'
      : n.neighbourhood_cleansed,
    occupancy: Math.round((n.avg_occupancy || 0) * 100),
    avgPrice: Math.round(n.avg_price || 0),
  }));

  return (
    <div className="chart-card">
      <div className="card-header">Occupancy by Neighborhood</div>
      <div className="card-body">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data} layout="vertical" margin={{ left: 10, right: 20 }}>
            <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} />
            <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 12 }} />
            <Tooltip formatter={(v) => `${v}%`} />
            <Bar dataKey="occupancy" fill="#16a34a" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
