import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { formatCurrency } from '../../utils/formatters';

export default function NeighborhoodChart({ data = [] }) {
  const chartData = data.slice(0, 12).map(n => ({
    name: n.neighbourhood_cleansed?.length > 18
      ? n.neighbourhood_cleansed.slice(0, 16) + '...'
      : n.neighbourhood_cleansed,
    revenue: Math.round(n.avg_monthly_revenue || 0),
  }));

  return (
    <div className="chart-card">
      <div className="card-header">Top Neighborhoods by Revenue</div>
      <div className="card-body">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 20 }}>
            <XAxis type="number" tickFormatter={v => formatCurrency(v)} />
            <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 12 }} />
            <Tooltip formatter={v => formatCurrency(v)} />
            <Bar dataKey="revenue" fill="#006AFF" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
