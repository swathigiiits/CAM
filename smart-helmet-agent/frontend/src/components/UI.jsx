import React from 'react'

export function ScoreRing({ score, size = 160, label = 'Attention Score' }) {
  const r = (size - 20) / 2
  const circ = 2 * Math.PI * r
  const fill = circ * (1 - score / 100)

  return (
    <div className="score-ring-container flex flex-col items-center gap-2">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <defs>
          <linearGradient id="scoreGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#0c8de8" />
            <stop offset="100%" stopColor="#36a7f7" />
          </linearGradient>
        </defs>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth="10" stroke="#e2e8f0" strokeLinecap="round" />
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none" strokeWidth="10"
          stroke="url(#scoreGrad)"
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={fill}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: 'stroke-dashoffset 1.5s cubic-bezier(0.4,0,0.2,1)' }}
        />
        <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle"
          className="font-display" fontSize={size * 0.2} fontWeight="700" fill="#0f172a">
          {score}
        </text>
        <text x="50%" y="65%" textAnchor="middle" dominantBaseline="middle" fontSize={size * 0.1} fill="#64748b">
          / 100
        </text>
      </svg>
      <p className="text-xs font-medium text-slate-500">{label}</p>
    </div>
  )
}

export function RiskBadge({ level }) {
  const cls = level === 'Low' ? 'risk-low' : level === 'Moderate' ? 'risk-moderate' : 'risk-high'
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold ${cls}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${
        level === 'Low' ? 'bg-green-500' : level === 'Moderate' ? 'bg-amber-500' : 'bg-red-500'
      }`} />
      {level} Risk
    </span>
  )
}

export function MetricCard({ icon: Icon, label, value, unit = '', trend }) {
  return (
    <div className="glass-card card-3d p-5 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="w-9 h-9 rounded-xl flex items-center justify-center bg-brand-50">
          {Icon && <Icon className="w-4 h-4 text-brand-500" />}
        </div>
        {trend !== undefined && (
          <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
            trend > 0 ? 'bg-green-50 text-green-700' :
            trend < 0 ? 'bg-red-50 text-red-600' : 'bg-slate-100 text-slate-500'
          }`}>
            {trend > 0 ? '↑' : trend < 0 ? '↓' : '—'} {Math.abs(trend)}{unit}
          </span>
        )}
      </div>
      <div>
        <p className="text-2xl font-display font-bold text-slate-900">
          {value}<span className="text-sm font-normal text-slate-400 ml-0.5">{unit}</span>
        </p>
        <p className="text-xs text-slate-500 mt-0.5">{label}</p>
      </div>
    </div>
  )
}

export function ProgressBar({ value, max = 100, label }) {
  const pct = Math.round(Math.min(100, (value / max) * 100))
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-xs">
        <span className="text-slate-600 font-medium">{label}</span>
        <span className="text-brand-600 font-mono font-semibold">{pct}%</span>
      </div>
      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
        <div className="progress-fill h-full" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export function GazeHeatmap({ grid, width = 320, height = 180 }) {
  if (!grid || !grid.length) {
    return <div className="flex items-center justify-center h-40 text-slate-400 text-sm">No heatmap data</div>
  }

  return (
    <div className="relative rounded-xl overflow-hidden border border-slate-200" style={{ width, height }}>
      <div className="absolute inset-0 bg-gradient-to-b from-sky-50 to-slate-100">
        <div className="absolute bottom-0 left-0 right-0 h-3/5 bg-gradient-to-b from-slate-200 to-slate-300" />
      </div>
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 50 50" preserveAspectRatio="none">
        {grid.map((row, ri) =>
          row.map((val, ci) => {
            if (val < 5) return null
            const alpha = val / 255
            const r = Math.round(255 * (alpha > 0.5 ? 1 : alpha * 2))
            const g = Math.round(180 * (alpha > 0.5 ? 2 - alpha * 2 : 1))
            return (
              <rect key={`${ri}-${ci}`} x={ci} y={ri} width={1} height={1}
                fill={`rgba(${r},${g},0,${Math.min(0.75, alpha)})`} />
            )
          })
        )}
      </svg>
      <div className="absolute bottom-1 right-2 text-xs text-slate-600 font-mono bg-white/80 px-1.5 py-0.5 rounded border border-slate-200">
        Gaze Density
      </div>
    </div>
  )
}

export function ShimmerCard({ h = 120 }) {
  return (
    <div className="glass-card p-5 overflow-hidden" style={{ height: h }}>
      <div className="shimmer h-full w-full rounded-lg" />
    </div>
  )
}

export function LevelBadge({ level }) {
  const styles = {
    Beginner: 'bg-slate-100 text-slate-700',
    Intermediate: 'bg-blue-50 text-blue-800',
    Advanced: 'bg-purple-50 text-purple-800',
  }
  return (
    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${styles[level] || styles.Beginner}`}>
      {level}
    </span>
  )
}

export function SceneDistribution({ data }) {
  if (!data) return null
  const items = [
    { key: 'road', label: 'Road', color: '#0c8de8' },
    { key: 'vehicles', label: 'Vehicles', color: '#7bc5fb' },
    { key: 'mirrors', label: 'Mirrors', color: '#36a7f7' },
    { key: 'traffic_signs', label: 'Traffic Signs', color: '#0057a1' },
    { key: 'traffic_lights', label: 'Traffic Lights', color: '#ca8a04' },
    { key: 'pedestrians', label: 'Pedestrians', color: '#d97706' },
    { key: 'dashboard', label: 'Dashboard', color: '#94a3b8' },
    { key: 'distractors', label: 'Distractors', color: '#ef4444' },
  ]

  return (
    <div className="space-y-2.5">
      {items.map(({ key, label, color }) => {
        const val = data[key] || 0
        return (
          <div key={key} className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-slate-600">{label}</span>
              <span className="font-mono font-medium" style={{ color }}>{val}%</span>
            </div>
            <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
              <div className="h-full rounded-full transition-all duration-700"
                style={{ width: `${val}%`, background: color }} />
            </div>
          </div>
        )
      })}
    </div>
  )
}

export function EmptyState({ icon: Icon, title, subtitle, action }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-4 text-center">
      {Icon && (
        <div className="w-16 h-16 rounded-2xl bg-brand-50 flex items-center justify-center">
          <Icon className="w-8 h-8 text-brand-400" />
        </div>
      )}
      <div>
        <h3 className="font-display font-semibold text-slate-800 text-lg">{title}</h3>
        {subtitle && <p className="text-slate-500 text-sm mt-1">{subtitle}</p>}
      </div>
      {action}
    </div>
  )
}
