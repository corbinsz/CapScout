import axios from 'axios';
import demoApi from './demoApi';

/** Live scoring only when both are set. The Vercel preview leaves them unset and uses the Austin sample. */
export const LIVE_API = import.meta.env.VITE_USE_LIVE_API === '1' ? (import.meta.env.VITE_API_URL || '') : '';
export const API_BASE = LIVE_API || '';

const liveApi = axios.create({
  baseURL: LIVE_API,
  timeout: 120000,
  withCredentials: true,
  headers: { 'Content-Type': 'application/json' },
});

function isColdStart(error) {
  const status = error.response?.status;
  return error.code === 'ECONNABORTED' || error.code === 'ERR_NETWORK' || status === 502 || status === 503;
}

// One quiet retry when a live server is still waking up.
liveApi.interceptors.response.use(null, async (error) => {
  const config = error.config;
  if (config && isColdStart(error) && !config._retried) {
    config._retried = true;
    await new Promise((resolve) => setTimeout(resolve, 2500));
    return liveApi(config);
  }
  return Promise.reject(error);
});

const api = LIVE_API ? liveApi : demoApi;

/**
 * Resolve a backend-relative URL (e.g. /api/streetview/...) to an absolute
 * URL when the frontend and backend are on different domains.
 */
export function resolveBackendUrl(path) {
  if (!path || !path.startsWith('/api/')) return path;
  if (!LIVE_API) return path;
  const origin = LIVE_API.replace(/\/api\/?$/, '');
  return origin ? `${origin}${path}` : path;
}

export function friendlyNetworkMessage(error) {
  if (isColdStart(error)) {
    return 'The scoring server is waking up. Wait a few seconds and try again.';
  }
  return 'We could not load that just now. Try again in a moment.';
}

export default api;
