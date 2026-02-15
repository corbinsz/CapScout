import api from './api';

export async function fetchAnalytics(city) {
  const { data } = await api.get('/analytics', { params: { city } });
  return data;
}
