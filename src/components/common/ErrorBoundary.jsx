import { Component } from 'react'
import { reportError } from '../../monitoring'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { failed: false }
  }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch(error) {
    reportError(error)
  }

  render() {
    if (this.state.failed) {
      return (
        <div className="legal-page">
          <h1>Something broke on this page</h1>
          <p>Refresh the page. If it keeps happening, the server may still be waking up.</p>
          <button className="btn btn-primary" type="button" onClick={() => window.location.reload()}>
            Refresh
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
