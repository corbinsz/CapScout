import { useEffect, useState } from 'react';
import api from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';

export default function PaywallModal() {
  const { paywallOpen, closePaywall, openAuth, user, refresh } = useAuth();
  const [status, setStatus] = useState(null);
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!paywallOpen) return;
    api.get('/billing/status').then(res => setStatus(res.data)).catch(() => setStatus(null));
  }, [paywallOpen]);

  if (!paywallOpen) return null;

  const checkout = async () => {
    if (!user) {
      closePaywall();
      openAuth();
      return;
    }
    setBusy(true);
    setMessage('');
    try {
      const res = await api.post('/billing/checkout');
      if (res.data.url) {
        window.location.assign(res.data.url);
        return;
      }
      setMessage(res.data.message || 'Checkout is not connected on this server yet.');
    } catch (err) {
      setMessage(err.response?.data?.error || 'Checkout could not start.');
    } finally {
      setBusy(false);
    }
  };

  const devActivate = async () => {
    setBusy(true);
    try {
      await api.post('/billing/dev-activate');
      await refresh();
      closePaywall();
    } catch (err) {
      setMessage(err.response?.data?.error || 'Dev upgrade is turned off on this server.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={closePaywall} role="presentation">
      <div className="modal-card" onClick={(event) => event.stopPropagation()} role="dialog" aria-labelledby="paywall-title">
        <button className="modal-close" onClick={closePaywall} aria-label="Close">×</button>
        <p className="eyebrow">CapScout Pro</p>
        <h2 id="paywall-title">Top deals report</h2>
        <p className="modal-copy">
          Pro downloads a CSV of the highest-scoring homes in the city you are viewing,
          with the cash-flow line and the plain-English reason for each score.
          Browsing scores stays free.
        </p>
        <ul className="paywall-list">
          <li>Ranked Top deals export for the selected city</li>
          <li>Cap rate, cash flow, and confidence on every row</li>
          <li>{status?.price_label || '$29/month'} when Stripe is connected</li>
        </ul>
        <button className="btn btn-primary" type="button" onClick={checkout} disabled={busy}>
          {user ? 'Continue to checkout' : 'Sign in to upgrade'}
        </button>
        {message && <p className="form-note">{message}</p>}
        {import.meta.env.DEV && (
          <button className="text-button" type="button" onClick={devActivate}>
            Developer: mark this account Pro without a charge
          </button>
        )}
        <p className="fine-print">
          No card is charged unless the server has a real Stripe secret key. This build ships with none.
        </p>
      </div>
    </div>
  );
}
