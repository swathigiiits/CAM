import React, { useEffect, useState } from 'react'
import { getSessions, getProgress } from '../utils/api.js'
import { TrendingUp, Award, AlertCircle } from 'lucide-react'
import { LevelBadge, RiskBadge, EmptyState, ShimmerCard } from '../components/UI.jsx'
import { PageHeader } from '../components/layout/PageHeader.jsx'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, ReferenceLine
} from 'recharts'

const LEVEL_ORDER = { Beginner: 1, Intermediate: 2, Advanced: 3 }

export default function Progress() {
  const [driverId, setDriverId] = useState('anonymous')
  const [progress, setProgress] = useState(null)
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    getSessions().then(s => {
      setSessions(s)
      const ids = [...new Set(s.map(x => x.driver_id))]
      if (ids.length > 0) setDriverId(ids[0])
    })
  }, [])

  useEffect(() => {
    if (!driverId) return
    setLoading(true)
    setError('')
    getProgress(driverId)
      .then(setProgress)
      .catch(e => setError(e.response?.data?.detail || 'No sessions found'))
      .finally(() => setLoading(false))
  }, [driverId])

  const driverIds = [...new Set(sessions.map(s => s.driver_id))]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <PageHeader title="Progress Tracking" subtitle="Longitudinal attention improvement" />
        <div className="flex items-center gap-2 shrink-0">
          <label className="text-xs font-medium text-slate-500">Driver ID:</label>
          <select value={driverId} onChange={e => setDriverId(e.target.value)}
            className="px-3 py-2 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-brand-400 bg-white">
            {driverIds.map(id => <option key={id} value={id}>{id}</option>)}
          </select>
        </div>
      </div>

      {loading && (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-4">{[...Array(3)].map((_, i) => <ShimmerCard key={i} h={100} />)}</div>
          <ShimmerCard h={280} />
        </div>
      )}

      {error && !loading && (
        <div className="glass-card p-12">
          <EmptyState icon={TrendingUp} title="No progress data" subtitle={error} />
        </div>
      )}

      {progress && !loading && (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
            <div className="glass-card card-3d p-6 text-center">
              <p className="text-3xl font-display font-bold text-slate-900">{progress.total_sessions}</p>
              <p className="text-sm text-slate-500 mt-1">Total Sessions</p>
            </div>
            <div className="glass-card card-3d p-6 text-center">
              <div className="flex justify-center mb-2">
                <LevelBadge level={progress.current_level} />
              </div>
              <p className="text-sm text-slate-500">Current Level</p>
            </div>
            <div className="glass-card card-3d p-6 text-center">
              <p className={`text-3xl font-display font-bold ${
                progress.overall_improvement > 0 ? 'text-green-600' :
                progress.overall_improvement < 0 ? 'text-red-500' : 'text-slate-500'
              }`}>
                {progress.overall_improvement > 0 ? '+' : ''}{progress.overall_improvement}
              </p>
              <p className="text-sm text-slate-500 mt-1">Score Change</p>
            </div>
            <div className="glass-card card-3d p-6 text-center">
              <p className="text-3xl font-display font-bold text-slate-900">{progress.behavioral_consistency ?? '—'}%</p>
              <p className="text-sm text-slate-500 mt-1">Consistency</p>
            </div>
          </div>

          {(progress.weekly_trends?.length > 0) && (
            <div className="glass-card p-6">
              <h2 className="font-display font-semibold text-slate-800 mb-4">Weekly Progress</h2>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={progress.weekly_trends}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="week" tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: 'white', border: '1px solid #e2e8f0', borderRadius: 12 }} />
                  <Line type="monotone" dataKey="avg_score" name="Avg Score" stroke="#0c8de8" strokeWidth={2} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {(progress.attention_maturity_trend?.length > 0) && (
            <div className="glass-card p-6">
              <h2 className="font-display font-semibold text-slate-800 mb-4">Attention Maturity Trend</h2>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={progress.attention_maturity_trend}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="session_num" tickFormatter={v => `S${v}`} tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: 'white', border: '1px solid #e2e8f0', borderRadius: 12 }} />
                  <Line type="monotone" dataKey="maturity" name="Maturity" stroke="#7c3aed" strokeWidth={2} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
              {progress.risk_reduction_trend != null && (
                <p className="text-xs text-slate-500 mt-2 text-center">
                  Risk reduction trend: {progress.risk_reduction_trend > 0 ? '+' : ''}{progress.risk_reduction_trend}%
                </p>
              )}
            </div>
          )}

          {/* Multi-metric chart */}
          <div className="glass-card p-6">
            <h2 className="font-display font-semibold text-slate-800 mb-4">Score Progression</h2>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={progress.sessions}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="session_num" tickFormatter={v => `S${v}`}
                  tick={{ fontSize: 12, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 12, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: 'white', border: '1px solid #e2e8f0', borderRadius: 12 }}
                  labelFormatter={v => `Session ${v}`}
                />
                <Legend />
                <ReferenceLine y={75} stroke="#22c55e" strokeDasharray="4 3"
                  label={{ value: 'Advanced', fill: '#22c55e', fontSize: 10 }} />
                <ReferenceLine y={55} stroke="#f59e0b" strokeDasharray="4 3"
                  label={{ value: 'Intermediate', fill: '#f59e0b', fontSize: 10 }} />
                <Line type="monotone" dataKey="attention_score" name="Attention Score"
                  stroke="#0c8de8" strokeWidth={2.5} dot={{ fill: '#0c8de8', r: 5 }} />
                <Line type="monotone" dataKey="road_attention_pct" name="Road Attention %"
                  stroke="#36a7f7" strokeWidth={1.5} strokeDasharray="4 3" dot={{ fill: '#36a7f7', r: 3 }} />
                <Line type="monotone" dataKey="hazard_awareness" name="Hazard Awareness"
                  stroke="#7bc5fb" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Distraction trend */}
          <div className="glass-card p-6">
            <h2 className="font-display font-semibold text-slate-800 mb-4">Distraction Events (lower is better)</h2>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={progress.sessions}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="session_num" tickFormatter={v => `S${v}`}
                  tick={{ fontSize: 12, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 12, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: 'white', border: '1px solid #e2e8f0', borderRadius: 12 }} labelFormatter={v => `Session ${v}`} />
                <Line type="monotone" dataKey="distraction_events" name="Distraction Events"
                  stroke="#ef4444" strokeWidth={2} dot={{ fill: '#ef4444', r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Session table */}
          <div className="glass-card p-6 overflow-x-auto">
            <h2 className="font-display font-semibold text-slate-800 mb-4">Session History</h2>
            <table className="w-full text-sm min-w-[600px]">
              <thead>
                <tr className="border-b border-slate-100">
                  {['#', 'Date', 'Score', 'Road%', 'Maturity', 'Reaction', 'Distractions', 'Risk'].map(h => (
                    <th key={h} className="text-left py-2 px-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {progress.sessions.map((s) => (
                  <tr key={s.session_id} className="border-b border-slate-50 hover:bg-brand-50 transition-colors">
                    <td className="py-2.5 px-3 font-mono text-xs text-slate-400">{s.session_num}</td>
                    <td className="py-2.5 px-3 text-xs text-slate-600">{new Date(s.timestamp).toLocaleDateString()}</td>
                    <td className="py-2.5 px-3 font-semibold text-slate-800">{s.attention_score}</td>
                    <td className="py-2.5 px-3 text-slate-600">{s.road_attention_pct}%</td>
                    <td className="py-2.5 px-3 text-slate-600">{s.attention_maturity ?? '—'}</td>
                    <td className="py-2.5 px-3 text-slate-600">{s.avg_hazard_reaction_time_s ? `${s.avg_hazard_reaction_time_s}s` : '—'}</td>
                    <td className="py-2.5 px-3 text-slate-600">{s.distraction_events}</td>
                    <td className="py-2.5 px-3"><RiskBadge level={s.risk_level} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
