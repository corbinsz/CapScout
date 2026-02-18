import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
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
