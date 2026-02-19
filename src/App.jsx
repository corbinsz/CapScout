import { lazy, Suspense } from 'react'
import { Routes, Route } from 'react-router-dom'
import { CityProvider } from './contexts/CityContext'
import Layout from './components/layout/Layout'
import LoadingSpinner from './components/common/LoadingSpinner'

const Dashboard = lazy(() => import('./pages/Dashboard'))
const MapPage = lazy(() => import('./pages/MapPage'))
const Analytics = lazy(() => import('./pages/Analytics'))
const PropertyPage = lazy(() => import('./pages/PropertyPage'))
const PredictionProcess = lazy(() => import('./pages/PredictionProcess'))
const TrainingData = lazy(() => import('./pages/TrainingData'))
const Settings = lazy(() => import('./pages/Settings'))

function App() {
  return (
    <CityProvider>
      <Suspense fallback={<LoadingSpinner message="Loading..." />}>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/map" element={<MapPage />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/property/:id" element={<PropertyPage />} />
            <Route path="/prediction-process" element={<PredictionProcess />} />
            <Route path="/training-data" element={<TrainingData />} />
            <Route path="/settings" element={<Settings />} />
          </Route>
        </Routes>
      </Suspense>
    </CityProvider>
  )
}

export default App
