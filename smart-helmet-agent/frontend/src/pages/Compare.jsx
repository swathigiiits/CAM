import React, { useEffect, useState } from 'react'
import { getSessions, compareSessions } from '../utils/api.js'
import { GitCompare, Plus, X, BarChart2 } from 'lucide-react'
import { RiskBadge, LevelBadge, EmptyState, ShimmerCard } from '../components/UI.jsx'
import { PageHeader } from '../components/layout/PageHeader.jsx'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip, Legend
} from 'recharts'

const COLORS = ['#0c8de8', '#36a7f7', '#7bc5fb', '#0057a1']

export default function Compare() {
  const [sessions, setSessions] = useState([])
  const [selected, setSelected] = useState([])
  const [result, setResult] = useState(null)
  const [deltas, setDeltas] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    getSessions().then(setSessions)
  }, [])

  const toggle = (id) => {
    setSelected(s => s.includes(id) ? s.filter(x => x !== id) : s.length < 4 ? [...s, id] : s)
  }

  const handleCompare = async () => {
    if (selected.length < 2) return
    setLoading(true)
    try {
      const r = await compareSessions(selected)
      setResult(r.comparison)
      setDeltas(r.deltas || null)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  const radarData = result ? [
    { metric: 'Attention', ...Object.fromEntries(result.map((s, i) => [`S${i + 1}`, s.attention_score])) },
    { metric: 'Road Focus', ...Object.fromEntries(result.map((s, i) => [`S${i + 1}`, s.road_attention_pct])) },
    { metric: 'Stability', ...Object.fromEntries(result.map((s, i) => [`S${i + 1}`, s.gaze_stability || 0])) },
    { metric: 'Hazard React', ...Object.fromEntries(result.map((s, i) => [`S${i + 1}`, Math.max(0, 100 - (s.avg_hazard_reaction_time_s || 0) * 80)])) },
    { metric: 'No Distraction', ...Object.fromEntries(result.map((s, i) => [`S${i + 1}`, Math.max(0, 100 - s.distraction_events * 10)])) },
  ] : []

  return (
    <div className="space-y-6">
      <PageHeader title="Session Comparison" subtitle="Compare up to 4 sessions side by side" />

      {sessions.length === 0 ? (
        <div className="glass-card p-12">
          <EmptyState icon={GitCompare} title="No sessions to compare" subtitle="Upload at least 2 sessions first" />
        </div>
      ) : (
        <>
          {/* Session selector */}
          <div className="glass-card p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="font-display font-semibold text-slate-800">Select Sessions ({selected.length}/4)</h2>
              {selected.length >= 2 && (
                <button onClick={handleCompare}
                  className="flex items-center gap-2 px-4 py-2 rounded-xl bg-brand-500 text-white text-sm font-semibold hover:bg-brand-600 transition-colors">
                  <BarChart2 className="w-4 h-4" /> Compare
                </button>
              )}
            </div>
            <div className="space-y-2">
              {sessions.map(s => (
                <div key={s.session_id}
                  onClick={() => toggle(s.session_id)}
                  className={`flex items-center gap-3 p-3.5 rounded-xl cursor-pointer transition-all duration-200 border-2 ${
                    selected.includes(s.session_id)
                      ? 'border-brand-400 bg-brand-50'
                      : 'border-transparent hover:border-brand-200 hover:bg-slate-50'
                  }`}
                >
                  <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center transition-colors ${
                    selected.includes(s.session_id) ? 'border-brand-500 bg-brand-500' : 'border-slate-300'
                  }`}>
                    {selected.includes(s.session_id) && <div className="w-2 h-2 rounded-full bg-white" />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs text-slate-500">#{s.session_id}</span>
                      <RiskBadge level={s.metrics.risk_level} />
                    </div>
                    <p className="text-xs text-slate-400 mt-0.5">
                      {new Date(s.timestamp).toLocaleDateString()} · {s.session_type} · Score: {s.metrics.attention_score}
                    </p>
                  </div>
                  <span className="font-display font-bold text-brand-600 text-lg">{s.metrics.attention_score}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Results */}
          {loading && <ShimmerCard h={300} />}

          {result && !loading && (
            <div className="space-y-6">
              {deltas && result.length >= 2 && (
                <div className="glass-card p-6">
                  <h2 className="font-display font-semibold text-slate-800 mb-3">
                    Session {result[result.length - 1].session_id} vs {result[0].session_id}
                  </h2>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    {[
                      ['Attention Score', deltas.attention_score_delta, '/100'],
                      ['Road Focus', deltas.road_focus_delta, '%'],
                      ['Distractions Reduced', deltas.distraction_delta, ''],
                      ['Hazard Reaction', deltas.hazard_reaction_delta, 's faster'],
                    ].map(([label, val, unit]) => (
                      <div key={label} className="metric-pill p-4 rounded-xl">
                        <p className="text-xs text-slate-500">{label}</p>
                        <p className={`text-lg font-bold ${val > 0 ? 'text-green-600' : val < 0 ? 'text-red-500' : 'text-slate-600'}`}>
                          {val > 0 ? '+' : ''}{val}{unit}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {/* Radar */}
              <div className="glass-card p-6">
                <h2 className="font-display font-semibold text-slate-800 mb-4">Radar Comparison</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="#e2e8f0" />
                    <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12, fill: '#64748b' }} />
                    <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                    {result.map((s, i) => (
                      <Radar key={s.session_id} name={`Session ${s.session_id}`} dataKey={`S${i + 1}`}
                        stroke={COLORS[i]} fill={COLORS[i]} fillOpacity={0.08} strokeWidth={2} />
                    ))}
                    <Tooltip contentStyle={{ background: 'white', border: '1px solid #e2e8f0', borderRadius: 12 }} />
                    <Legend />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              {/* Table comparison */}
              <div className="glass-card p-6 overflow-x-auto">
                <h2 className="font-display font-semibold text-slate-800 mb-4">Metrics Table</h2>
                <table className="w-full text-sm min-w-[500px]">
                  <thead>
                    <tr className="border-b border-slate-100">
                      <th className="text-left py-2 px-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">Metric</th>
                      {result.map((s, i) => (
                        <th key={s.session_id} className="text-center py-2 px-3 text-xs font-semibold uppercase tracking-wide"
                          style={{ color: COLORS[i] }}>
                          #{s.session_id}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ['Attention Score', r => r.attention_score, '/100'],
                      ['Road Attention', r => r.road_attention_pct, '%'],
                      ['Hazard Awareness', r => Math.round(r.hazard_awareness), '%'],
                      ['Hazard Response', r => r.avg_hazard_reaction_time_s || '—', 's'],
                      ['Env. Scanning', r => Math.round(r.environmental_scanning || 0), '%'],
                      ['Distraction Events', r => r.distraction_events, ''],
                      ['Dangerous Glances', r => r.dangerous_glances ?? 0, ''],
                      ['Risk Level', r => r.risk_level, ''],
                    ].map(([label, getter, unit]) => (
                      <tr key={label} className="border-b border-slate-50">
                        <td className="py-2.5 px-3 text-slate-600 font-medium text-xs">{label}</td>
                        {result.map((s, i) => {
                          const val = getter(s)
                          const vals = result.map(getter).filter(v => typeof v === 'number')
                          const best = vals.length === 0 ? false
                            : label === 'Distraction Events' || label === 'Hazard Response'
                            ? getter(s) === Math.min(...vals)
                            : label === 'Risk Level' ? false
                            : getter(s) === Math.max(...vals)
                          return (
                            <td key={s.session_id} className={`py-2.5 px-3 text-center font-mono text-sm font-semibold ${best ? 'text-green-600' : 'text-slate-700'}`}>
                              {val}{unit}
                              {best && <span className="ml-1 text-green-500 text-xs">★</span>}
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="text-xs text-slate-400 mt-3">★ = best value in category</p>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
