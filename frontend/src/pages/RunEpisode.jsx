import { useState, useEffect } from 'react'
import { PlayCircle, AlertCircle } from 'lucide-react'
import { getTasks, getModels, runEpisode } from '../api/client'
import { SOCChart, ArbitrageChart, FreqRegChart, PeakShavingChart, RewardChart } from '../components/EpisodeCharts'

export default function RunEpisode() {
  const [tasks, setTasks] = useState([])
  const [models, setModels] = useState([])
  const [form, setForm] = useState({ task: 'hard', model_name: '', seed: 42 })
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([getTasks(), getModels()]).then(([t, m]) => {
      setTasks(t); setModels(m)
      if (m.length) setForm(f => ({ ...f, model_name: m.find(x => x.includes('hard')) || m[0] }))
    })
  }, [])

  const onSubmit = async (e) => {
    e.preventDefault()
    setRunning(true); setError(null); setResult(null)
    try {
      const res = await runEpisode(form)
      setResult(res)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setRunning(false)
    }
  }

  return (
    <>
      <header className="page-header">
        <h1>Run Episode</h1>
        <p>Visualize step-by-step agent decisions on a single scenario.</p>
      </header>

      <div className="page-body">
        <div className="two-col">
          {/* Controls */}
          <div className="card">
            <div className="card-head">
              <span className="card-title">Configuration</span>
            </div>
            <div className="card-body">
              <form onSubmit={onSubmit} className="gap-14">
                <div className="form-group">
                  <label className="form-label">Task Complexity</label>
                  <select className="form-control" value={form.task} onChange={e => setForm({ ...form, task: e.target.value })}>
                    {tasks.map(t => <option key={t.id} value={t.id}>{t.label} ({t.id})</option>)}
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Trained Agent Model</label>
                  <select className="form-control" value={form.model_name} onChange={e => setForm({ ...form, model_name: e.target.value })}>
                    {models.map(m => <option key={m} value={m}>{m}</option>)}
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Scenario Seed</label>
                  <input type="number" className="form-control" value={form.seed} onChange={e => setForm({ ...form, seed: parseInt(e.target.value) || 0 })} />
                </div>
                <button type="submit" className="btn btn-primary" disabled={running || !form.model_name}>
                  {running ? <><div className="spinner" style={{width:14,height:14}}/> Running...</> : <><PlayCircle size={15}/> Execute Episode</>}
                </button>
              </form>
            </div>
          </div>

          {/* Results Area */}
          <div>
            {error && (
              <div className="alert alert-error flex gap-8">
                <AlertCircle size={16} /> <b>Error running simulation:</b> {error}
              </div>
            )}
            
            {running && (
              <div className="loading-box card" style={{ height: 400 }}>
                <div className="spinner" /> Environment step processing in progress...
              </div>
            )}

            {result && !running && (
              <>
                <div className="reward-total">
                  <div className="gap-8" style={{flex:1}}>
                    <span className="text-2 uppercase" style={{fontSize:11, letterSpacing:1}}>Total Episode Reward</span>
                    <span className="big">${result.total_reward.toLocaleString(undefined,{maximumFractionDigits:0})}</span>
                  </div>
                  <div className="gap-8 text-right">
                    <div className="badge badge-purple">{result.task.toUpperCase()}</div>
                    <div className="badge badge-cyan">{result.steps.length} Steps</div>
                  </div>
                </div>

                <div className="chart-grid">
                  <SOCChart data={result.steps} />
                  <ArbitrageChart data={result.steps} />
                  {['medium', 'hard'].includes(result.task) && <FreqRegChart data={result.steps} />}
                  {result.task === 'hard' && <PeakShavingChart data={result.steps} />}
                  <RewardChart data={result.steps} />
                </div>
              </>
            )}
            
            {!result && !running && !error && (
              <div className="hero" style={{ textAlign: 'center', padding: '60px 20px', background: 'transparent', borderStyle: 'dashed' }}>
                <PlayCircle size={40} className="text-3 mb-14" style={{margin:'0 auto'}} />
                <h3 style={{color:'var(--text-2)'}}>No simulation data</h3>
                <p>Configure the scenario on the left and click Execute to visualize.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}
