import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { formatCurrency } from '../../utils/formatters';

export default function BedroomBreakdownChart({ data = [] }) {
  const chartData = data.map(d => ({
    name: `${d.bedrooms} BR`,
    count: d.count,
    avgRevenue: Math.round(d.avg_revenue || 0),
  }));

  return (
    <div className="chart-card">
      <div className="card-header">Property Type Breakdown</div>
      <div className="card-body">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <XAxis dataKey="name" />
            <YAxis yAxisId="left" />
            <YAxis yAxisId="right" orientation="right" tickFormatter={v => formatCurrency(v)} />
            <Tooltip formatter={(v, name) => name === 'avgRevenue' ? formatCurrency(v) : v} />
            <Legend />
            <Bar yAxisId="left" dataKey="count" fill="#006AFF" name="Listings" radius={[4, 4, 0, 0]} />
            <Bar yAxisId="right" dataKey="avgRevenue" fill="#d9840d" name="Avg Revenue" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
