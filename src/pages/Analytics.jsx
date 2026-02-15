import useAnalytics from '../hooks/useAnalytics';
import { useCity } from '../contexts/CityContext';
import MetricCard from '../components/common/MetricCard';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';
import NeighborhoodChart from '../components/charts/NeighborhoodChart';
import PriceDistributionChart from '../components/charts/PriceDistributionChart';
import BedroomBreakdownChart from '../components/charts/BedroomBreakdownChart';
import RevenueChart from '../components/charts/RevenueChart';
import { formatCurrency, formatPercent, formatNumber } from '../utils/formatters';

export default function Analytics() {
  const { data, loading, error } = useAnalytics();
  const { cityMeta } = useCity();
  const cityDisplay = cityMeta?.display || 'Austin, TX';

  if (loading) return <LoadingSpinner message="Loading analytics..." />;
  if (error) return <ErrorMessage message={error} />;
  if (!data) return null;

  return (
    <div>
      <div className="page-header">
        <h1>Market Analytics</h1>
        <p>{cityDisplay} short-term rental market overview and insights</p>
      </div>

      <div className="metrics-row">
        <MetricCard label="Total Listings" value={formatNumber(data.total_listings)} />
        <MetricCard label="Avg Nightly Rate" value={formatCurrency(data.avg_nightly_rate)} />
        <MetricCard label="Median Nightly Rate" value={formatCurrency(data.median_nightly_rate)} />
        <MetricCard label="Avg Occupancy" value={formatPercent(data.avg_occupancy)} />
        <MetricCard label="Avg Monthly Revenue" value={formatCurrency(data.avg_monthly_revenue)} />
        <MetricCard label="Avg Rating" value={data.avg_rating?.toFixed(2)} />
      </div>

      <h2 className="section-divider">Market Breakdown</h2>

      <div className="chart-grid">
        <NeighborhoodChart data={data.neighborhoods} />
        <RevenueChart neighborhoods={data.neighborhoods} />
        <PriceDistributionChart distribution={data.price_distribution} />
        <BedroomBreakdownChart data={data.bedroom_breakdown} />
      </div>
    </div>
  );
}
