import { lazy, Suspense } from 'react'
import { Routes, Route } from 'react-router-dom'
import { CityProvider } from './contexts/CityContext'
import { FavoritesProvider } from './contexts/FavoritesContext'
import { AuthProvider } from './contexts/AuthContext'
import Layout from './components/layout/Layout'
import LoadingSpinner from './components/common/LoadingSpinner'
import Home from './pages/Home'

const Dashboard = lazy(() => import('./pages/Dashboard'))
const MapPage = lazy(() => import('./pages/MapPage'))
const Analytics = lazy(() => import('./pages/Analytics'))
const PropertyPage = lazy(() => import('./pages/PropertyPage'))
const PredictionProcess = lazy(() => import('./pages/PredictionProcess'))
const TrainingData = lazy(() => import('./pages/TrainingData'))
const Settings = lazy(() => import('./pages/Settings'))
const Favorites = lazy(() => import('./pages/Favorites'))
const Compare = lazy(() => import('./pages/Compare'))
const Terms = lazy(() => import('./pages/Terms'))
const Privacy = lazy(() => import('./pages/Privacy'))
const Markets = lazy(() => import('./pages/Markets'))

function App() {
  return (
    <CityProvider>
      <AuthProvider>
      <FavoritesProvider>
        <Suspense fallback={<LoadingSpinner message="Loading..." />}>
          <Routes>
            <Route element={<Layout />}>
              <Route path="/" element={<Home />} />
              <Route path="/deals" element={<Dashboard />} />
              <Route path="/map" element={<MapPage />} />
              <Route path="/favorites" element={<Favorites />} />
              <Route path="/compare" element={<Compare />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/markets" element={<Markets />} />
              <Route path="/property/:id" element={<PropertyPage />} />
              <Route path="/prediction-process" element={<PredictionProcess />} />
              <Route path="/training-data" element={<TrainingData />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/terms" element={<Terms />} />
              <Route path="/privacy" element={<Privacy />} />
            </Route>
          </Routes>
        </Suspense>
      </FavoritesProvider>
      </AuthProvider>
    </CityProvider>
  )
}

export default App
