import { useState, useEffect } from 'react'
import { PlayCircle, AlertCircle, Sparkles } from 'lucide-react'
import { getTasks, getModels, runEvaluate, runLLM } from '../api/client'
import ScoreBar from '../components/ScoreBar'

export default function Evaluate() {
  const [tasks, setTasks] = useState([])
  const [models, setModels] = useState([])
  const [form, setForm] = useState({ task: 'hard', model_name: '', num_seeds: 10, seed_start: 300 })
  const [evaluating, setEvaluating] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  
  // LLM states
  const [llmLoading, setLlmLoading] = useState(false)
  const [llmResult, setLlmResult] = useState(null)

  useEffect(() => {
    Promise.all([getTasks(), getModels()]).then(([t, m]) => {
      setTasks(t); setModels(m)
      if (m.length) setForm(f => ({ ...f, model_name: m.find(x => x.includes('hard')) || m[0] }))
    })
  }, [])

  const onEval = async (e) => {
    e.preventDefault()
    setEvaluating(true); setError(null); setResult(null); setLlmResult(null)
    try {
      const res = await runEvaluate(form)
      setResult(res)
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setEvaluating(false)
    }
  }

  const triggerLLM = async () => {
    if (!result) return
    setLlmLoading(true)
    try {
      const llmRes = await runLLM(result)
      setLlmResult(llmRes)
    } catch (err) {
      console.error(err)
      setLlmResult({ error: "Failed to communicate with LLM API." })
    } finally {
      setLlmLoading(false)
    }
  }

  return (
    <>
      <header className="page-header">
        <h1>Evaluation Suite</h1>
        <p>Run multi-seed evaluation, calculate dimension scores, and request AI grading.</p>
      </header>

      <div className="page-body">
        <div className="two-col">
          <div className="card">
            <div className="card-head"><span className="card-title">Test Config</span></div>
            <div className="card-body">
              <form onSubmit={onEval} className="gap-14">
                <div className="form-group">
                  <label className="form-label">Task Context</label>
                  <select className="form-control" value={form.task} onChange={e => setForm({...form, task: e.target.value})}>
                    {tasks.map(t => <option key={t.id} value={t.id}>{t.label}</option>)}
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Agent Model</label>
                  <select className="form-control" value={form.model_name} onChange={e => setForm({...form, model_name: e.target.value})}>
                    {models.map(m => <option key={m} value={m}>{m}</option>)}
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Validation Seeds</label>
                  <input type="number" className="form-control" min={1} max={50} value={form.num_seeds} onChange={e => setForm({...form, num_seeds: parseInt(e.target.value)||10})} />
                  <div style={{fontSize:10, color:'var(--text-3)', marginTop:4}}>Starting at index {form.seed_start}</div>
                </div>
                <button className="btn btn-primary" type="submit" disabled={evaluating || !form.model_name}>
                  {evaluating ? <><div className="spinner" style={{width:14,height:14}}/> Simulating {form.num_seeds} envs...</> : 'Start Evaluation'}
                </button>
              </form>
            </div>
          </div>

          <div>
            {error && <div className="alert alert-error mb-14">{error}</div>}
            
            {evaluating && (
              <div className="card loading-box" style={{height: 400}}>
                <div className="spinner" /> Integrating multi-seed rollouts...
              </div>
            )}

            {result && !evaluating && (
              <div className="gap-14">
                <div className="chart-grid">
                  <div className="overall-hero" style={{gridRow: 'span 2'}}>
                    <div className="overall-num">{result.scores.overall.toFixed(3)}</div>
                    <div className="overall-label">Overall Readiness Score</div>
                    <div className="divider" style={{width: '60%'}}/>
                    <div className="row gap-14 text-2" style={{fontSize: 12}}>
                      <div>Tasks Tested: <b>{result.task.toUpperCase()}</b></div>
                      <div>Sample Size: <b>{result.num_seeds} Seeds</b></div>
                    </div>
                  </div>

                  <div className="card">
                    <div className="card-head"><span className="card-title">Component Scores</span></div>
                    <div className="card-body">
                      <ScoreBar scores={result.scores} />
                    </div>
                  </div>

                  <div className="card">
                    <div className="card-head"><span className="card-title">Raw Telemetry Means</span></div>
                    <div className="card-body" style={{fontSize:12}}>
                      <div className="row-sb mb-14"><span>Mean Reward</span><b className="mono text-green">${result.reward_mean.toFixed(0)}</b></div>
                      <div className="row-sb mb-14"><span>Reward StdDev</span><b className="mono">±${result.reward_std.toFixed(0)}</b></div>
                      <div className="row-sb mb-14"><span>Peak Shaving Misses</span><b className="mono">{result.peak_violation_pct.toFixed(1)}%</b></div>
                      <div className="row-sb mb-14"><span>Peak Hr SOC Baseline</span><b className="mono">{(result.soc_at_peak_mean*100).toFixed(1)}%</b></div>
                      <div className="row-sb"><span>Cycles Output/Ep</span><b className="mono text-amber">{result.avg_cycles_per_ep.toFixed(1)}</b></div>
                    </div>
                  </div>
                </div>

                {/* LLM Grading Panel */}
                <div className="card" style={{borderColor: 'rgba(99,102,241,.3)', borderTopWidth: 2}}>
                  <div className="card-head">
                    <span className="card-title row"><Sparkles size={16} className="text-accent"/> Gemini 2.5 AI Analysis</span>
                    {!llmResult && !llmLoading && (
                      <button className="btn btn-sm btn-ghost" onClick={triggerLLM} style={{color:'var(--accent)'}}>Request Grading</button>
                    )}
                  </div>
                  <div className="card-body">
                    {llmLoading ? (
                      <div className="loading-box" style={{padding: '30px 0'}}><div className="spinner"/> Requesting Gemini assessment...</div>
                    ) : llmResult ? (
                      llmResult.available ? (
                        <div className="llm-panel">
                          <div className="llm-verdict">{llmResult.verdict}</div>
                          <p style={{fontSize: 14, marginBottom: 16}}>{llmResult.summary}</p>
                          <div className="chart-grid">
                            <div>
                              <div className="llm-section text-green">Strengths</div>
                              <ul className="llm-ul mb-14">{llmResult.strengths?.map((s,i) => <li key={i}>{s}</li>)}</ul>
                            </div>
                            <div>
                              <div className="llm-section text-rose">Weaknesses</div>
                              <ul className="llm-ul">{llmResult.weaknesses?.map((w,i) => <li key={i} style={{color: 'var(--text-3)'}}>{w}</li>)}</ul>
                            </div>
                          </div>
                          <div className="llm-section text-amber">Recommendations</div>
                          <ul className="llm-ul mb-14">{llmResult.recommendations?.map((r,i) => <li key={i}>{r}</li>)}</ul>
                          <div className="llm-section">Detailed Analysis ({llmResult.confidence} Confidence)</div>
                          <div className="llm-detail">{llmResult.detailed_analysis}</div>
                        </div>
                      ) : (
                        <div className="alert alert-warn">{llmResult.error || llmResult.summary}</div>
                      )
                    ) : (
                      <div style={{color:'var(--text-3)', fontSize:13, fontStyle:'italic'}}>AI analysis not requested yet. Click "Request Grading" above.</div>
                    )}
                  </div>
                </div>

              </div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}
