import { NavLink } from 'react-router-dom';

export default function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <div className="navbar-brand">
          <NavLink to="/">
            <span className="brand-icon">S</span>
            STR Predictor
          </NavLink>
        </div>
        <div className="navbar-links">
          <NavLink to="/" end>Dashboard</NavLink>
          <NavLink to="/map">Map</NavLink>
          <NavLink to="/analytics">Analytics</NavLink>
          <NavLink to="/settings">Settings</NavLink>
        </div>
      </div>
    </nav>
  );
}
