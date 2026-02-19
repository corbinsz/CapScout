import api from './api';

export async function fetchProperties(params = {}, opts = {}) {
  const { data } = await api.get('/properties', { params, signal: opts.signal });
  return data;
}

export async function fetchProperty(id, city, overrides = {}) {
  const { data } = await api.get(`/properties/${id}`, { params: { city, ...overrides } });
  return data;
}

export async function searchProperties(query, city) {
  const { data } = await api.get('/search', { params: { q: query, city } });
  return data.results || [];
}

export async function predictProperty(propertyData) {
  const { data } = await api.post('/predict', propertyData);
  return data;
}

export async function fetchFeatureImportance(id, city) {
  const { data } = await api.get(`/properties/${id}/feature-importance`, { params: { city } });
  return data;
}
