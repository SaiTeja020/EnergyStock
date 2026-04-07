import {
  ComposedChart, AreaChart, Area, LineChart, Line, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'

const DARK_TICK  = { fill: '#475569', fontSize: 11 }
const GRID_PROPS = { stroke: 'rgba(255,255,255,0.05)', strokeDasharray: '3 3' }

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#0d1530', border: '1px solid rgba(255,255,255,.1)',
      borderRadius: 8, padding: '10px 14px', fontSize: 12,
    }}>
      <div style={{ color: '#94a3b8', marginBottom: 6 }}>Step {label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color, marginBottom: 2 }}>
          {p.name}: <strong>{typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</strong>
        </div>
      ))}
    </div>
  )
}

/* ── 1. Battery SOC ─────────────────────────────────────── */
export function SOCChart({ data }) {
  return (
    <div className="chart-card">
      <div className="chart-title">Battery State of Charge</div>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data}>
          <CartesianGrid {...GRID_PROPS} />
          <XAxis dataKey="step" tick={DARK_TICK} tickLine={false} interval="preserveStartEnd" label={{ value: 'Step', fill: '#475569', fontSize: 11, position: 'insideBottomRight', offset: -4 }} />
          <YAxis domain={[0, 1]} tick={DARK_TICK} tickLine={false} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0.2} stroke="#f59e0b" strokeDasharray="4 3" strokeWidth={1.5} label={{ value: '20% Min', fill: '#f59e0b', fontSize: 10 }} />
          <Area type="monotone" dataKey="soc" name="SOC" stroke="#6366f1" fill="rgba(99,102,241,0.18)" strokeWidth={2} dot={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ── 2. Energy Arbitrage (LMP vs action) ────────────────── */
export function ArbitrageChart({ data }) {
  return (
    <div className="chart-card">
      <div className="chart-title">Energy Arbitrage</div>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data}>
          <CartesianGrid {...GRID_PROPS} />
          <XAxis dataKey="step" tick={DARK_TICK} tickLine={false} interval="preserveStartEnd" />
          <YAxis yAxisId="lmp" tick={DARK_TICK} tickLine={false} label={{ value: 'LMP ($/MWh)', angle: -90, fill: '#f59e0b', fontSize: 10, position: 'insideLeft' }} />
          <YAxis yAxisId="act" orientation="right" domain={[-1.2, 1.2]} tick={DARK_TICK} tickLine={false} label={{ value: 'Action', angle: 90, fill: '#10b981', fontSize: 10, position: 'insideRight' }} />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
          <Line yAxisId="lmp" type="monotone" dataKey="lmp" name="LMP" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
          <Bar  yAxisId="act" dataKey="action_ea" name="EA Action" fill="rgba(16,185,129,0.55)" />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ── 3. Frequency Regulation ────────────────────────────── */
export function FreqRegChart({ data }) {
  return (
    <div className="chart-card">
      <div className="chart-title">Frequency Regulation</div>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data}>
          <CartesianGrid {...GRID_PROPS} />
          <XAxis dataKey="step" tick={DARK_TICK} tickLine={false} interval="preserveStartEnd" />
          <YAxis yAxisId="rew" tick={DARK_TICK} tickLine={false} label={{ value: 'FR Reward ($)', angle: -90, fill: '#a78bfa', fontSize: 10, position: 'insideLeft' }} />
          <YAxis yAxisId="act" orientation="right" domain={[-1.2, 1.2]} tick={DARK_TICK} tickLine={false} label={{ value: 'Action', angle: 90, fill: '#94a3b8', fontSize: 10, position: 'insideRight' }} />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
          <Area yAxisId="rew" type="monotone" dataKey="r_fr" name="FR Reward" stroke="#a78bfa" fill="rgba(167,139,250,0.15)" strokeWidth={2} dot={false} />
          <Line yAxisId="act" type="monotone" dataKey="action_fr" name="FR Action" stroke="#94a3b8" strokeWidth={1} dot={false} strokeDasharray="4 2" />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ── 4. Peak Shaving ─────────────────────────────────────── */
export function PeakShavingChart({ data }) {
  return (
    <div className="chart-card">
      <div className="chart-title">Peak Shaving</div>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data}>
          <CartesianGrid {...GRID_PROPS} />
          <XAxis dataKey="step" tick={DARK_TICK} tickLine={false} interval="preserveStartEnd" />
          <YAxis tick={DARK_TICK} tickLine={false} label={{ value: 'MW', angle: -90, fill: '#475569', fontSize: 10, position: 'insideLeft' }} />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />
          <ReferenceLine y={20} stroke="#f43f5e" strokeDasharray="4 3" strokeWidth={1.5} label={{ value: 'Peak Limit', fill: '#f43f5e', fontSize: 10 }} />
          <Area type="monotone" dataKey="baseline_load" name="Baseline Load" stroke="#f43f5e" fill="rgba(244,63,94,0.12)" strokeWidth={1.5} dot={false} />
          <Line type="monotone" dataKey="net_load" name="Net Load" stroke="#22d3ee" strokeWidth={2} dot={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ── 5. Cumulative Reward ────────────────────────────────── */
export function RewardChart({ data }) {
  let cum = 0
  const cumData = data.map(d => ({ ...d, cum_reward: (cum += d.reward) }))
  return (
    <div className="chart-card">
      <div className="chart-title">Cumulative Reward</div>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={cumData}>
          <CartesianGrid {...GRID_PROPS} />
          <XAxis dataKey="step" tick={DARK_TICK} tickLine={false} interval="preserveStartEnd" />
          <YAxis tick={DARK_TICK} tickLine={false} tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} />
          <Tooltip content={<CustomTooltip />} />
          <Area type="monotone" dataKey="cum_reward" name="Cumulative Reward" stroke="#10b981" fill="rgba(16,185,129,0.15)" strokeWidth={2} dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
