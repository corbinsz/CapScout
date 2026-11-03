import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';
import Footer from './Footer';
import AuthModal from '../auth/AuthModal';
import PaywallModal from '../auth/PaywallModal';

export default function Layout() {
  return (
    <div className="app-layout">
      <a className="skip-link" href="#main">Skip to content</a>
      <Navbar />
      <main id="main" className="app-main">
        <Outlet />
      </main>
      <Footer />
      <AuthModal />
      <PaywallModal />
    </div>
  );
}
