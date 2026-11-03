import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import Logo from './Logo';
import SearchBar from './SearchBar';
import { useAuth } from '../../contexts/AuthContext';
import api from '../../services/api';

const SEARCH_PAGES = ['/deals', '/map'];

export default function Navbar() {
  const { pathname } = useLocation();
  const showSearch = SEARCH_PAGES.includes(pathname);
  const [menuOpen, setMenuOpen] = useState(false);
  const { user, openAuth, setUser, isPro } = useAuth();

  const signOut = async () => {
    await api.post('/auth/logout');
    setUser(null);
    setMenuOpen(false);
  };

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <div className="navbar-brand">
          <NavLink to="/" onClick={() => setMenuOpen(false)}>
            <Logo size={30} />
            CapScout
          </NavLink>
        </div>
        {showSearch && <SearchBar />}
        <button
          className={`navbar-hamburger${menuOpen ? ' open' : ''}`}
          onClick={() => setMenuOpen(v => !v)}
          aria-label="Toggle menu"
          aria-expanded={menuOpen}
        >
          <span /><span /><span />
        </button>
        <div className={`navbar-links${menuOpen ? ' navbar-links--open' : ''}`}>
          <NavLink to="/deals" onClick={() => setMenuOpen(false)}>Deals</NavLink>
          <NavLink to="/markets" onClick={() => setMenuOpen(false)}>Cities</NavLink>
          <NavLink to="/map" onClick={() => setMenuOpen(false)}>Map</NavLink>
          <NavLink to="/favorites" onClick={() => setMenuOpen(false)}>Saved</NavLink>
          {user ? (
            <button className="nav-account" type="button" onClick={signOut}>
              {isPro ? 'Pro' : 'Free'} · Sign out
            </button>
          ) : (
            <button className="nav-account" type="button" onClick={() => { setMenuOpen(false); openAuth(); }}>
              Sign in
            </button>
          )}
        </div>
      </div>
      {menuOpen && <div className="navbar-backdrop" onClick={() => setMenuOpen(false)} />}
    </nav>
  );
}
