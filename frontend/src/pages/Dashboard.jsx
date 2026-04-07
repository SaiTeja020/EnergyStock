import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { PlayCircle, BarChart3, Database, Zap } from 'lucide-react'
import { getTasks, getModels } from '../api/client'

export default function Dashboard() {
  const [tasks, setTasks] = useState([])
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getTasks(), getModels()])
      .then(([t, m]) => {
        setTasks(t)
        setModels(m)
      })
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="page-body loading-box">
        <div className="spinner" /> Loading dashboard...
      </div>
    )
  }

  return (
    <div className="page-body">
      <div className="hero">
        <h2>Battery Energy Storage System RL</h2>
        <p>
          Evaluate and visualize reinforcement learning agents trained for grid-scale energy co-optimization.
        </p>
      </div>

      <div className="metric-grid mb-14">
        <div className="metric-card">
          <div className="metric-label">Available Models</div>
          <div className="metric-value">{models.length}</div>
          <div className="metric-sub text-accent">found in train/models</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Tasks Configured</div>
          <div className="metric-value">{tasks.length}</div>
          <div className="metric-sub text-amber">Easy, Medium, Hard</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Data Points</div>
          <div className="metric-value">720</div>
          <div className="metric-sub text-green">Hours per simulation</div>
        </div>
      </div>

      <div className="chart-grid">
        <div className="card">
          <div className="card-head">
            <span className="card-title row"><PlayCircle size={16} className="text-accent" /> Run Episode</span>
          </div>
          <div className="card-body gap-14">
            <p style={{ color: 'var(--text-2)', fontSize: 13 }}>
              Run a trained agent on a single generated PJM scenario. Visualize Battery SOC, LMP (Arbitrage), Regulation Signals, and Load (Peak Shaving) step-by-step.
            </p>
            <Link to="/run" className="btn btn-primary" style={{ width: 'fit-content' }}>Run Simulation</Link>
          </div>
        </div>

        <div className="card">
          <div className="card-head">
            <span className="card-title row"><BarChart3 size={16} className="text-cyan" /> Evaluate Logic</span>
          </div>
          <div className="card-body gap-14">
            <p style={{ color: 'var(--text-2)', fontSize: 13 }}>
              Perform rigorous multi-seed evaluation testing generalization on unseen scenarios. Features Gemini 2.5 Flash LLM analysis for automated readiness grading.
            </p>
            <Link to="/evaluate" className="btn btn-ghost" style={{ width: 'fit-content' }}>Evaluate Model</Link>
          </div>
        </div>
      </div>
    </div>
  )
}
