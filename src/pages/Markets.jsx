import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import api from '../services/api'
import { useCity } from '../contexts/CityContext'
import { formatCurrency } from '../utils/formatters'
import LoadingSpinner from '../components/common/LoadingSpinner'
import ErrorMessage from '../components/common/ErrorMessage'

export default function Markets() {
  const { switchCity } = useCity()
  const [rows, setRows] = useState(null)
  const [note, setNote] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    document.title = 'CapScout — Compare cities'
    api.get('/markets/leaders')
      .then((res) => {
        setRows(res.data.items || [])
        setNote(res.data.note || '')
      })
      .catch((err) => setError(err.response?.data?.error || err.message))
  }, [])

  if (error) return <ErrorMessage message={error} />
  if (!rows) return <LoadingSpinner message="Scoring the top home in each city…" />

  return (
    <section>
      <div className="page-header">
        <h1>Where the best screen is</h1>
        <p>The highest CapScout score between $150k and $750k in each city, with the same loan math.</p>
      </div>
      {note && <p className="source-note">{note}</p>}
      <div className="market-table-wrap">
        <table className="market-table">
          <thead>
            <tr>
              <th scope="col">City</th>
              <th scope="col">Top score</th>
              <th scope="col">Cash flow / mo</th>
              <th scope="col">Home</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.city}>
                <td>{row.display}</td>
                <td>{row.top_score == null ? '—' : row.top_score.toFixed(1)}</td>
                <td>{formatCurrency(row.monthly_cashflow)}</td>
                <td>
                  {row.property_id ? (
                    <Link to={`/property/${row.property_id}?city=${row.city}`} onClick={() => switchCity(row.city)}>
                      {row.address || 'View home'}
                    </Link>
                  ) : (
                    row.message || 'No scored homes'
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
