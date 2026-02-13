import api from './api';

export async function fetchProperties(params = {}) {
  const { data } = await api.get('/properties', { params });
  return data;
}

export async function fetchProperty(id) {
  const { data } = await api.get(`/properties/${id}`);
  return data;
}

export async function predictProperty(propertyData) {
  const { data } = await api.post('/predict', propertyData);
  return data;
}
