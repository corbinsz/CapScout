import { API_BASE } from './services/api'

/** First-party counts only, and only when the owner opts in at build time. */
export function track(name) {
  if (import.meta.env.VITE_ANALYTICS !== '1') return
  const body = JSON.stringify({ name })
  const url = `${API_BASE}/events`
  if (typeof navigator !== 'undefined' && navigator.sendBeacon) {
    navigator.sendBeacon(url, new Blob([body], { type: 'application/json' }))
    return
  }
  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
    credentials: 'include',
    keepalive: true,
  }).catch(() => {})
}
