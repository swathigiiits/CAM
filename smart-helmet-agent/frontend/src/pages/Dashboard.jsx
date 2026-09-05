import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getSessions } from '../utils/api.js'
import { Eye, AlertTriangle, TrendingUp, Upload, ChevronRight, Activity } from 'lucide-react'
import {
  ScoreRing, RiskBadge, MetricCard, LevelBadge, EmptyState, ShimmerCard
} from '../components/UI.jsx'
import { PageHeader, PrimaryButton, MotionSection } from '../components/layout/PageHeader.jsx'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'
import frontVideo from '../assets/front.mp4'

export default function Dashboard() {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getSessions().then(setSessions).catch(console.error).finally(() => setLoading(false))
  }, [])

  const latest = sessions[0]
  const avgScore = sessions.length
    ? Math.round(sessions.reduce((a, s) => a + s.metrics.attention_score, 0) / sessions.length)
    : 0

  const chartData = sessions.slice().reverse().map((s, i) => ({
    name: `S${i + 1}`,
    score: s.metrics.attention_score,
    road: s.metrics.road_attention_pct,
  }))

  return (
    <div className="relative min-h-screen">
      <video 
        autoPlay 
        loop 
        muted 
        className="absolute top-0 left-0 w-full h-full object-cover -z-10"
      >
        <source src={frontVideo} type="video/mp4" />
      </video>
      <div className="absolute top-0 left-0 w-full h-full bg-black/30 -z-10"></div>
      <div className="space-y-8 p-8">
        <PageHeader
          title="Dashboard"
          subtitle="Driver attention intelligence — gaze and video analytics"
          action={
            <PrimaryButton to="/upload">
              <Upload className="w-4 h-4" /> New Session
            </PrimaryButton>
          }
        />

        <MotionSection className="glass-card p-6 md:p-8">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <p className="text-xs font-mono text-brand-600 uppercase tracking-widest mb-1">Attention Analytics</p>
              <p className="font-display text-2xl md:text-3xl font-bold text-slate-900">
                {sessions.length ? `${sessions.length} sessions analyzed` : 'Upload your first session'}
              </p>
              <p className="text-slate-600 text-sm mt-1 max-w-md">
                Quantify road focus, hazard response, and attention maturity over time.
              </p>
            </div>
            {latest && (
              <div className="metric-pill px-5 py-3 inline-flex items-center gap-3 self-start md:self-auto">
                <span className="font-display text-3xl font-bold text-brand-600">{Math.round(latest.metrics.attention_score)}</span>
                <div>
                  <p className="text-xs text-slate-500">Latest score</p>
                  <RiskBadge level={latest.metrics.risk_level} />
                </div>
              </div>
            )}
          </div>
        </MotionSection>

        {loading ? (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => <ShimmerCard key={i} h={100} />)}
          </div>
        ) : sessions.length === 0 ? (
          <MotionSection delay={0.1} className="glass-card p-12">
            <EmptyState
              icon={Activity}
              title="No sessions yet"
              subtitle="Upload gaze CSV and optional driving video to get started"
              action={
                <PrimaryButton to="/upload">
                  <Upload className="w-4 h-4" /> Upload Session
                </PrimaryButton>
              }
            />
          </MotionSection>
        ) : (
          <>
            <MotionSection delay={0.05} className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard icon={Activity} label="Total Sessions" value={sessions.length} />
              <MetricCard icon={Eye} label="Avg Attention Score" value={avgScore} unit="/100" />
              <MetricCard icon={TrendingUp} label="Latest Score" value={latest?.metrics.attention_score || 0} unit="/100" />
              <MetricCard icon={AlertTriangle} label="Latest Distractions" value={latest?.metrics.distraction_events || 0} />
            </MotionSection>

            <div className="grid lg:grid-cols-3 gap-6">
              {latest && (
                <MotionSection delay={0.1} className="glass-card card-3d p-6 flex flex-col items-center gap-4">
                  <div className="w-full flex items-center justify-between">
                    <h2 className="font-display font-semibold text-slate-800">Latest Session</h2>
                    <span className="text-xs font-mono text-slate-400">#{latest.session_id}</span>
                  </div>
                  <ScoreRing score={Math.round(latest.metrics.attention_score)} />
                  <div className="flex gap-2 flex-wrap justify-center">
                    <RiskBadge level={latest.metrics.risk_level} />
                    <LevelBadge level={latest.metrics.driver_level} />
                  </div>
                  <div className="w-full grid grid-cols-2 gap-3 text-sm">
                    {[
                      ['Road Attention', `${latest.metrics.road_attention_pct}%`],
                      ['Stability', `${Math.round(latest.metrics.gaze_stability)}%`],
                      ['Distractions', `${latest.metrics.distraction_events}`],
                      ['Duration', `${latest.metrics.session_duration_s}s`],
                    ].map(([k, v]) => (
                      <div key={k} className="metric-pill rounded-xl p-3">
                        <p className="text-xs text-slate-500">{k}</p>
                        <p className="font-semibold text-slate-800 font-mono">{v}</p>
                      </div>
                    ))}
                  </div>
                  <Link to={`/sessions/${latest.session_id}`}
                    className="w-full text-center py-2.5 rounded-xl border border-brand-200 text-brand-700 text-sm font-semibold hover:bg-brand-50 transition-colors flex items-center justify-center gap-1 pointer-events-auto">
                    View Details <ChevronRight className="w-3.5 h-3.5" />
                  </Link>
                </MotionSection>
              )}

              <MotionSection delay={0.15} className="glass-card p-6 lg:col-span-2">
                <h2 className="font-display font-semibold text-slate-800 mb-4">Score Trend</h2>
                {chartData.length >= 2 ? (
                  <ResponsiveContainer width="100%" height={220}>
                    <AreaChart data={chartData}>
                      <defs>
                        <linearGradient id="scoreGradChart" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#0c8de8" stopOpacity={0.2} />
                          <stop offset="95%" stopColor="#0c8de8" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="name" tick={{ fontSize: 12, fill: '#64748b' }} axisLine={false} tickLine={false} />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 12, fill: '#64748b' }} axisLine={false} tickLine={false} />
                      <Tooltip
                        contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 12 }}
                        labelStyle={{ fontWeight: 600, color: '#0f172a' }}
                      />
                      <Area type="monotone" dataKey="score" stroke="#0c8de8" strokeWidth={2.5}
                        fill="url(#scoreGradChart)" dot={{ fill: '#0c8de8', r: 5 }} name="Attention Score" />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-52 flex items-center justify-center text-slate-500 text-sm">
                    Upload more sessions to see trends
                  </div>
                )}
              </MotionSection>
            </div>

            <MotionSection delay={0.2} className="glass-card p-6">
              <h2 className="font-display font-semibold text-slate-800 mb-4">Recent Sessions</h2>
              <div className="space-y-2">
                {sessions.slice(0, 8).map((s) => (
                  <Link key={s.session_id} to={`/sessions/${s.session_id}`}
                    className="flex items-center gap-4 p-3.5 rounded-xl hover:bg-brand-50/80 transition-colors group border border-transparent hover:border-brand-100">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-400 to-brand-600 flex items-center justify-center text-white font-display font-bold text-sm shadow">
                      {Math.round(s.metrics.attention_score)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs text-slate-500">#{s.session_id}</span>
                        <RiskBadge level={s.metrics.risk_level} />
                      </div>
                      <p className="text-xs text-slate-500 mt-0.5">
                        {new Date(s.timestamp).toLocaleString()} · {s.session_type}
                      </p>
                    </div>
                    <div className="text-right hidden sm:block">
                      <p className="text-sm font-semibold text-slate-700">{s.metrics.road_attention_pct}% road</p>
                      <p className="text-xs text-slate-500">{s.metrics.distraction_events} distractions</p>
                    </div>
                    <ChevronRight className="w-4 h-4 text-slate-400 group-hover:text-brand-500 transition-colors" />
                  </Link>
                ))}
              </div>
            </MotionSection>
          </>
        )}
      </div>
    </div>
  );
}
