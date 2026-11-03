import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import api from '../services/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(false);
  const [authOpen, setAuthOpen] = useState(false);
  const [paywallOpen, setPaywallOpen] = useState(false);

  useEffect(() => {
    api.get('/auth/me')
      .then(res => setUser(res.data.user || null))
      .catch(() => setUser(null))
      .finally(() => setReady(true));
  }, []);

  const refresh = useCallback(async () => {
    const res = await api.get('/auth/me');
    setUser(res.data.user || null);
    return res.data.user || null;
  }, []);

  const value = {
    user,
    ready,
    isPro: user?.tier === 'pro',
    authOpen,
    paywallOpen,
    openAuth: () => setAuthOpen(true),
    closeAuth: () => setAuthOpen(false),
    openPaywall: () => setPaywallOpen(true),
    closePaywall: () => setPaywallOpen(false),
    setUser,
    refresh,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
