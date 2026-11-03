import axios from 'axios';

export const API_BASE = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' },
});

// Retry once on timeout (handles Render cold starts)
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

/**
 * Resolve a backend-relative URL (e.g. /api/streetview/...) to an absolute
 * URL when the frontend and backend are on different domains.
 */
export function resolveBackendUrl(path) {
  if (!path || !path.startsWith('/api/')) return path;
  // API_BASE is e.g. "https://capscout.onrender.com/api" or "/api"
  const origin = API_BASE.replace(/\/api\/?$/, '');
  return origin ? `${origin}${path}` : path;
}

export default api;
