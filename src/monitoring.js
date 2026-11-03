import { API_BASE } from './services/api'

/** Reports a short error string when VITE_ERROR_REPORTING=1. Drops anything with an email. */
export function reportError(error) {
  if (import.meta.env.VITE_ERROR_REPORTING !== '1') return
  const message = String(error && error.message ? error.message : error || 'unknown').slice(0, 200)
  if (message.includes('@')) return
  const path = typeof location !== 'undefined' ? location.pathname : ''
  fetch(`${API_BASE}/client-errors`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ message, path }),
  }).catch(() => {})
}
