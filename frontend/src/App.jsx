import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { LayoutDashboard, PlayCircle, BarChart3, Zap } from 'lucide-react'
import Dashboard   from './pages/Dashboard'
import RunEpisode  from './pages/RunEpisode'
import Evaluate    from './pages/Evaluate'
import { getHealth } from './api/client'

const NAV = [
  { to: '/',          label: 'Dashboard',   icon: LayoutDashboard },
  { to: '/run',       label: 'Run Episode', icon: PlayCircle },
  { to: '/evaluate',  label: 'Evaluate',    icon: BarChart3 },
]

export default function App() {
  const [online, setOnline] = useState(null)
  const location = useLocation()

  useEffect(() => {
    getHealth()
      .then(() => setOnline(true))
      .catch(() => setOnline(false))
  }, [location.pathname])

  return (
    <div className="layout">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">⚡</div>
          <div>
            <div className="brand-title">BESS-RL</div>
            <div className="brand-sub">RL PLATFORM</div>
          </div>
        </div>

        <nav className="nav">
          <div className="nav-label">Navigation</div>
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
            >
              <Icon size={17} />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <span className={`status-dot${online === false ? ' offline' : ''}`} />
          {online === null ? 'Connecting…' : online ? 'Backend online' : 'Backend offline'}
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="main">
        <Routes>
          <Route path="/"         element={<Dashboard />} />
          <Route path="/run"      element={<RunEpisode />} />
          <Route path="/evaluate" element={<Evaluate />} />
        </Routes>
      </main>
    </div>
  )
}
