import { useRef, useState } from 'react';
import api from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';

export default function AuthModal() {
  const { authOpen, closeAuth, setUser } = useAuth();
  const [mode, setMode] = useState('register');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const honeypot = useRef(null);

  if (!authOpen) return null;

  const submit = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      const path = mode === 'register' ? '/auth/register' : '/auth/login';
      const res = await api.post(path, { email, password, company: honeypot.current?.value || '' });
      setUser(res.data.user);
      setPassword('');
      closeAuth();
    } catch (err) {
      setError(err.response?.data?.error || 'Something went wrong. Please try again.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={closeAuth} role="presentation">
      <div className="modal-card" onClick={(event) => event.stopPropagation()} role="dialog" aria-labelledby="auth-title">
        <button className="modal-close" onClick={closeAuth} aria-label="Close">×</button>
        <p className="eyebrow">Free account</p>
        <h2 id="auth-title">{mode === 'register' ? 'Save your research' : 'Welcome back'}</h2>
        <p className="modal-copy">
          An account keeps your email on this device&apos;s session and unlocks the path to the
          Top deals report. We store a password hash, not the password itself.
        </p>
        <form onSubmit={submit} className="stack-form">
          <label>
            Email
            <input
              type="email"
              autoComplete="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
              maxLength={254}
            />
          </label>
          <label>
            Password
            <input
              type="password"
              autoComplete={mode === 'register' ? 'new-password' : 'current-password'}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
              minLength={10}
              maxLength={128}
            />
          </label>
          {/* Hidden from people. Bots that fill every field get rejected. */}
          <input className="hp-field" tabIndex={-1} autoComplete="off" name="company" ref={honeypot} aria-hidden="true" />
          {error && <p className="form-error">{error}</p>}
          <button className="btn btn-primary" type="submit" disabled={busy}>
            {busy ? 'Working…' : mode === 'register' ? 'Create free account' : 'Sign in'}
          </button>
        </form>
        <button
          className="text-button"
          type="button"
          onClick={() => { setMode(mode === 'register' ? 'login' : 'register'); setError(''); }}
        >
          {mode === 'register' ? 'Already have an account? Sign in' : 'Need an account? Create one'}
        </button>
      </div>
    </div>
  );
}
