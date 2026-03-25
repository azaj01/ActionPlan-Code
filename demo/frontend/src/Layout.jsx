import React, { useState, useEffect } from 'react'

const API_BASE = import.meta.env.VITE_BACKEND_URL ||
  (import.meta.env.DEV ? 'http://localhost:8000' : 'https://8000-01kj30ae5r0yty705g9javjw46.cloudspaces.litng.ai')

function Layout({ children }) {
  const [isConnected, setIsConnected] = useState(false)

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/health?t=${Date.now()}`, { cache: 'no-store' })
        const data = await res.json()
        setIsConnected(data.sampler_loaded)
      } catch {
        setIsConnected(false)
      }
    }
    check()
    const id = setInterval(check, 5000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="app-container">
      <header className="header">
        <h1>
          <a href="/" className="header-logo">
            <img src="/actionplan.png" alt="ActionPlan" className="header-icon" />
          </a>
          <span>ActionPlan</span>
          <span className="header-link-sep"></span>
        </h1>
        <div className="status-indicator">
          <div className={`status-dot ${isConnected ? 'connected' : ''}`} />
          {isConnected ? 'Connected' : 'Disconnected'}
        </div>
      </header>
      {children}
    </div>
  )
}

export default Layout
