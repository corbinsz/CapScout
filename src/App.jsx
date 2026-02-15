import { Routes, Route } from 'react-router-dom'
import { CityProvider } from './contexts/CityContext'
import Layout from './components/layout/Layout'
import Dashboard from './pages/Dashboard'
import MapPage from './pages/MapPage'
import Analytics from './pages/Analytics'
import PropertyPage from './pages/PropertyPage'
import Settings from './pages/Settings'

function App() {
  return (
    <CityProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/map" element={<MapPage />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/property/:id" element={<PropertyPage />} />
          <Route path="/settings" element={<Settings />} />
        </Route>
      </Routes>
    </CityProvider>
  )
}

export default App
