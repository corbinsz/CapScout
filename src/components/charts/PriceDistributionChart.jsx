import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { formatCurrency } from '../../utils/formatters';

export default function PriceDistributionChart({ distribution }) {
  if (!distribution) return null;

  const data = [
    { label: '10th %ile', value: distribution.p10 },
    { label: '25th %ile', value: distribution.p25 },
    { label: 'Median', value: distribution.p50 },
    { label: '75th %ile', value: distribution.p75 },
    { label: '90th %ile', value: distribution.p90 },
  ];

  return (
    <div className="chart-card">
      <div className="card-header">Nightly Rate Distribution</div>
      <div className="card-body">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <XAxis dataKey="label" tick={{ fontSize: 12 }} />
            <YAxis tickFormatter={v => formatCurrency(v)} />
            <Tooltip formatter={v => formatCurrency(v)} />
            <Bar dataKey="value" fill="#16a34a" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
