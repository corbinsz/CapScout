import { useState } from 'react'
import api from '../../services/api'
import { track } from '../../analytics'

export default function AlertForm({ city, cityName }) {
  const [email, setEmail] = useState('')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (event) => {
    event.preventDefault()
    setBusy(true)
    setError('')
    setMessage('')
    try {
      const res = await api.post('/alerts', { email, city, min_score: 70, cadence: 'weekly' })
      setMessage(res.data.message)
      setEmail('')
      track('alert_saved')
    } catch (err) {
      setError(err.response?.data?.error || 'Could not save that alert.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <form className="alert-form" onSubmit={submit}>
      <h2>Weekly alert for {cityName}</h2>
      <p>Email me when a home here scores 70 or higher. Nothing is sent until email delivery is connected.</p>
      <div className="alert-row">
        <label className="sr-only" htmlFor={`alert-${city}`}>Email</label>
        <input
          id={`alert-${city}`}
          type="email"
          required
          autoComplete="email"
          placeholder="you@example.com"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          maxLength={254}
        />
        <button className="btn btn-primary" type="submit" disabled={busy}>
          {busy ? 'Saving…' : 'Save alert'}
        </button>
      </div>
      {message && <p className="form-note">{message}</p>}
      {error && <p className="form-error">{error}</p>}
    </form>
  )
}
