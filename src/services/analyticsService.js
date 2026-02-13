import api from './api';

export async function fetchAnalytics() {
  const { data } = await api.get('/analytics');
  return data;
}
