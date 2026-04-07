const SCORE_COLORS = {
  reward:           '#6366f1',
  soc_readiness:    '#22d3ee',
  ps_adherence:     '#f59e0b',
  cycle_discipline: '#10b981',
  arb_accuracy:     '#a78bfa',
  consistency:      '#f43f5e',
}

const LABELS = {
  reward:           'Reward',
  soc_readiness:    'SOC Readiness',
  ps_adherence:     'Peak Shaving',
  cycle_discipline: 'Cycle Discipline',
  arb_accuracy:     'Arb. Accuracy',
  consistency:      'Consistency',
}

export default function ScoreBreakdown({ scores }) {
  if (!scores) return null
  const keys = Object.keys(LABELS)

  return (
    <div>
      {keys.map(k => {
        const val = scores[k] ?? 0
        const color = SCORE_COLORS[k] || '#6366f1'
        return (
          <div className="score-row" key={k}>
            <span className="score-label">{LABELS[k]}</span>
            <div className="score-track">
              <div
                className="score-fill"
                style={{ width: `${Math.max(0, Math.min(1, val)) * 100}%`, background: color }}
              />
            </div>
            <span className="score-val" style={{ color }}>{val.toFixed(2)}</span>
          </div>
        )
      })}
    </div>
  )
}
