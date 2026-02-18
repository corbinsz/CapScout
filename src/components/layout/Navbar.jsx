import { NavLink } from 'react-router-dom';
import Logo from './Logo';
import CitySelector from './CitySelector';
import SearchBar from './SearchBar';

export default function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <div className="navbar-brand">
          <NavLink to="/">
            <Logo size={30} />
            CapScout
          </NavLink>
          <CitySelector />
        </div>
        <SearchBar />
        <div className="navbar-links">
          <NavLink to="/" end>Dashboard</NavLink>
          <NavLink to="/map">Map</NavLink>
          <NavLink to="/analytics">Analytics</NavLink>
          <NavLink to="/prediction-process">Prediction Process</NavLink>
          <NavLink to="/settings">Settings</NavLink>
        </div>
      </div>
    </nav>
  );
}
