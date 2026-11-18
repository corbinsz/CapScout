import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
  headers: { 'Content-Type': 'application/json' },
});

api.interceptors.response.use(null, async (error) => {
  const config = error.config;
  if (
    error.code === 'ECONNABORTED' &&
    !config._retried
  ) {
    config._retried = true;
    return api(config);
  }
  return Promise.reject(error);
});

/** Resolve a backend-relative URL when the API base is absolute. */
export function resolveBackendUrl(path) {
  if (!path || !path.startsWith('/api/')) return path;
  const origin = API_BASE.replace(/\/api\/?$/, '');
  return origin ? `${origin}${path}` : path;
}

export default api;
