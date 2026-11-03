import { Link } from 'react-router-dom';

export const DISCLAIMER = 'CapScout scores are model estimates for research. They are not financial, investment, tax, or legal advice. Predicted rent and occupancy can be wrong. Confirm permits, insurance, taxes, and expenses before you buy.';

export default function Footer() {
  return (
    <footer className="site-footer">
      <p className="disclaimer">{DISCLAIMER}</p>
      <nav className="footer-links">
        <Link to="/deals">Deals</Link>
        <Link to="/analytics">Market stats</Link>
        <Link to="/settings">Data sources</Link>
        <Link to="/prediction-process">How scoring works</Link>
        <Link to="/terms">Terms</Link>
        <Link to="/privacy">Privacy</Link>
      </nav>
    </footer>
  );
}
