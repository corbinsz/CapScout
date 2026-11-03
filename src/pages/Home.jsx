import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useCity } from '../contexts/CityContext'
import { MARKETS } from '../utils/markets'
import { track } from '../analytics'
import AlertForm from '../components/alerts/AlertForm'

export default function Home() {
  const navigate = useNavigate()
  const { switchCity, currentCity } = useCity()

  useEffect(() => {
    document.title = 'CapScout — Can the rent pay for the house?'
    track('page_view')
  }, [])

  const openCity = (key) => {
    switchCity(key)
    track('city_selected')
    navigate('/deals')
  }

  const current = MARKETS.find((market) => market.key === currentCity) || MARKETS[0]

  return (
    <section className="landing">
      <p className="eyebrow">14 US cities · no account needed</p>
      <h1>Can the rent pay for the house?</h1>
      <p className="landing-lead">
        CapScout scores homes for sale as short-term rentals. You see what is left after the
        mortgage, taxes, and operating costs. Not a nightly rate on its own.
      </p>
      <div className="hero-actions">
        <button className="btn btn-primary" type="button" onClick={() => openCity(current.key)}>
          See top deals in {current.name}
        </button>
        <a className="btn btn-ghost" href="#cities">Choose another city</a>
      </div>
      <ul className="proof-row">
        <li>Same score in every city, so you can compare markets</li>
        <li>Plain-language reason on every home</li>
        <li>Estimates, not financial advice</li>
      </ul>
      <h2 id="cities">Pick a city</h2>
      <ul className="city-grid">
        {MARKETS.map((market) => (
          <li key={market.key}>
            <button type="button" onClick={() => openCity(market.key)}>
              <span className="city-name">{market.name}</span>
              <span className="city-state">{market.state}</span>
            </button>
          </li>
        ))}
      </ul>
      <AlertForm city={current.key} cityName={current.name} />
    </section>
  )
}
