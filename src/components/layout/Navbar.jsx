import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import Logo from './Logo';
import SearchBar from './SearchBar';

const SEARCH_PAGES = ['/', '/map'];

export default function Navbar() {
  const { pathname } = useLocation();
  const showSearch = SEARCH_PAGES.includes(pathname);
  const [menuOpen, setMenuOpen] = useState(false);

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
        >
          <span /><span /><span />
        </button>
        <div className={`navbar-links${menuOpen ? ' navbar-links--open' : ''}`}>
          <NavLink to="/" end onClick={() => setMenuOpen(false)}>Dashboard</NavLink>
          <NavLink to="/map" onClick={() => setMenuOpen(false)}>Map</NavLink>
          <NavLink to="/favorites" onClick={() => setMenuOpen(false)}>Favorites</NavLink>
          <NavLink to="/analytics" onClick={() => setMenuOpen(false)}>Analytics</NavLink>
          <NavLink to="/settings" onClick={() => setMenuOpen(false)}>Settings</NavLink>
        </div>
      </div>
      {menuOpen && <div className="navbar-backdrop" onClick={() => setMenuOpen(false)} />}
    </nav>
  );
}
