import React, { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getSession, setCalibration } from '../utils/api.js'
import { VideoGazePlayer, TemporalHeatmap } from '../components/VideoGazePlayer.jsx'
import {
  ScoreRing, RiskBadge, LevelBadge, ProgressBar,
  GazeHeatmap, SceneDistribution, ShimmerCard, MetricCard
} from '../components/UI.jsx'
import {
  Eye, Zap, AlertTriangle, Activity, Clock, ArrowLeft, Video, Target, Scan
} from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine
} from 'recharts'

function GazeOverlayFallback({ samples, width = 480, height = 270 }) {
  if (!samples?.length) return null
  return (
    <div className="relative bg-gradient-to-b from-sky-100 to-slate-300 rounded-xl overflow-hidden" style={{ width, height }}>
      <div className="absolute inset-0">
        <div className="absolute bottom-0 left-0 right-0 h-3/5 bg-gradient-to-b from-slate-400 to-slate-500" />
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-1 h-3/5"
          style={{ background: 'repeating-linear-gradient(to bottom, #fde047 0px, #fde047 15px, transparent 15px, transparent 30px)' }} />
      </div>
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1 1" preserveAspectRatio="none">
        {samples.map((pt, i) => (
          <circle key={i} cx={pt.x} cy={pt.y} r={0.006}
            fill={`rgba(12,141,232,${0.3 + pt.p * 0.4})`} stroke="rgba(12,141,232,0.6)" strokeWidth={0.001} />
        ))}
      </svg>
      <div className="absolute top-2 left-2 text-xs font-mono bg-white/90 text-slate-700 px-2 py-1 rounded-lg border border-slate-200">
        Gaze Overlay (no video)
      </div>
    </div>
  )
}

export default function SessionDetail() {
  const { id } = useParams()
  const [session, setSession] = useState(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState('overview')
  const [calibrating, setCalibrating] = useState(false)

  const reload = () => getSession(id).then(setSession)

  useEffect(() => {
    reload().catch(console.error).finally(() => setLoading(false))
  }, [id])

  const applyDefaultCalibration = async () => {
    setCalibrating(true)
    try {
      await setCalibration(id,
        [[0, 0], [1920, 0], [1920, 1080], [0, 1080]],
        [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]]
      )
      await reload()
    } catch (e) {
      console.error(e)
    } finally {
      setCalibrating(false)
    }
  }

  if (loading) return (
    <div className="space-y-4">
      <ShimmerCard h={80} />
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => <ShimmerCard key={i} h={100} />)}
      </div>
      <ShimmerCard h={300} />
    </div>
  )

  if (!session) return <div className="glass-card p-12 text-center text-slate-400">Session not found</div>

  const m = session.metrics
  const timeline = m.attention_timeline || []
  const TABS = ['overview', 'video', 'heatmap', 'timeline', 'coaching']

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Link to="/" className="p-2 rounded-xl hover:bg-slate-100 transition-colors">
          <ArrowLeft className="w-5 h-5 text-slate-500" />
        </Link>
        <div>
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="font-display font-bold text-xl text-slate-900">Session #{session.session_id}</h1>
            <RiskBadge level={m.risk_level} />
            <LevelBadge level={m.driver_level} />
            {m.scene_source === 'video_cv' && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-green-50 text-green-700 font-medium">Scene AI</span>
            )}
          </div>
          <p className="text-slate-500 text-xs mt-0.5">
            {new Date(session.timestamp).toLocaleString()} · {session.session_type} · {session.weather}
            {session.driver_id !== 'anonymous' && ` · Driver: ${session.driver_id}`}
            {session.speed_kmh != null && ` · ${session.speed_kmh} km/h`}
            {session.gps?.lat != null && ` · GPS ${session.gps.lat.toFixed(4)}, ${session.gps.lng.toFixed(4)}`}
          </p>
        </div>
      </div>

      <div className="flex gap-1 p-1 bg-white/80 border border-slate-200 rounded-xl w-fit flex-wrap shadow-sm">
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-lg text-sm font-medium capitalize transition-all duration-200 ${
              tab === t ? 'bg-brand-50 text-brand-800 shadow-sm border border-brand-100' : 'text-slate-600 hover:text-slate-800'
            }`}>
            {t}
          </button>
        ))}
      </div>

      {tab === 'overview' && (
        <div className="space-y-6">
          <div className="grid lg:grid-cols-4 gap-6">
            <div className="glass-card p-6 flex flex-col items-center gap-4 lg:col-span-1">
              <ScoreRing score={Math.round(m.attention_score)} size={160} />
              <div className="text-center space-y-0.5">
                <p className="text-xs text-slate-500">Maturity: <span className="font-semibold">{m.attention_maturity}/100</span></p>
                <p className="text-xs text-slate-500">Fixations: <span className="font-semibold">{m.fixation_count}</span></p>
                <p className="text-xs text-slate-500">Avg Fixation: <span className="font-semibold">{m.avg_fixation_duration_ms}ms</span></p>
                <p className="text-xs text-slate-500">Duration: <span className="font-semibold">{m.session_duration_s}s</span></p>
              </div>
            </div>
            <div className="glass-card p-6 lg:col-span-3 space-y-4">
              <h3 className="font-display font-semibold text-slate-800">Attention Breakdown</h3>
              <div className="space-y-3">
                <ProgressBar value={m.road_attention_pct} label="Road Attention" />
                <ProgressBar value={m.gaze_stability} label="Gaze Stability" />
                <ProgressBar value={m.hazard_awareness} label="Hazard Awareness" />
                <ProgressBar value={m.environmental_scanning || 0} label="Environmental Scanning" />
                <ProgressBar value={m.mirror_checks_pct} max={15} label="Mirror Checks" />
                <ProgressBar value={Math.max(0, 100 - m.distraction_events * 10)} label="Distraction Resistance" />
              </div>
            </div>
          </div>

          <div className="grid lg:grid-cols-2 gap-6">
            <div className="glass-card p-6">
              <h3 className="font-display font-semibold text-slate-800 mb-4">Scene-Aware Attention</h3>
              <SceneDistribution data={m.scene_distribution} />
              <p className="text-xs text-slate-400 mt-3">Source: {m.scene_source === 'video_cv' ? 'Video CV analysis' : 'Gaze zone mapping'}</p>
            </div>
            <div className="glass-card p-6 space-y-4">
              <h3 className="font-display font-semibold text-slate-800">Safety Metrics</h3>
              <div className="grid grid-cols-2 gap-3">
                <MetricCard icon={AlertTriangle} label="Off-Road Glances" value={m.off_road_glances ?? 0} />
                <MetricCard icon={Clock} label="Distraction Events" value={m.distraction_events} />
                <MetricCard icon={Target} label="Hazard Reaction" value={m.avg_hazard_reaction_time_s || '—'} unit={m.avg_hazard_reaction_time_s ? 's' : ''} />
                <MetricCard icon={Scan} label="Dangerous Glances" value={m.dangerous_glances ?? 0} />
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <MetricCard icon={Eye} label="Road Attention" value={m.road_attention_pct} unit="%" />
            <MetricCard icon={Zap} label="Gaze Stability" value={Math.round(m.gaze_stability)} unit="%" />
            <MetricCard icon={Activity} label="Env. Scanning" value={Math.round(m.environmental_scanning || 0)} unit="%" />
            <MetricCard icon={Activity} label="Gaze Points" value={m.total_gaze_points} />
          </div>

          {session.comparison && (
            <div className="glass-card p-6">
              <h3 className="font-display font-semibold text-slate-800 mb-4">vs. Previous Session</h3>
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
                {[
                  ['Attention Score', m.attention_score, session.comparison.attention_score, '/100'],
                  ['Road Focus', m.road_attention_pct, session.comparison.road_attention_pct, '%'],
                  ['Stability', Math.round(m.gaze_stability), Math.round(session.comparison.gaze_stability), '%'],
                  ['Distractions', m.distraction_events, session.comparison.distraction_events, ''],
                  ['Hazard Response', m.avg_hazard_reaction_time_s || 0, session.comparison.avg_hazard_reaction_time_s || 0, 's'],
                ].map(([label, curr, prev, unit]) => {
                  const delta = curr - prev
                  const better = label === 'Distractions' || label === 'Hazard Response' ? delta <= 0 : delta >= 0
                  return (
                    <div key={label} className="metric-pill rounded-xl p-4">
                      <p className="text-xs text-slate-500 mb-1">{label}</p>
                      <p className="font-display font-bold text-slate-900 text-lg">{curr}{unit}</p>
                      <p className={`text-xs font-semibold mt-1 ${better ? 'text-green-600' : 'text-red-500'}`}>
                        {delta > 0 ? '↑' : delta < 0 ? '↓' : '—'} {Math.abs(Number(delta.toFixed?.(2) ?? delta))}{unit} vs prev
                      </p>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {tab === 'video' && (
        <div className="space-y-6">
          <div className="glass-card p-6 space-y-4">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <h3 className="font-display font-semibold text-slate-800 flex items-center gap-2">
                <Video className="w-5 h-5 text-brand-500" /> Video + Gaze Sync
              </h3>
              {!session.calibration && (
                <button onClick={applyDefaultCalibration} disabled={calibrating}
                  className="text-xs px-3 py-1.5 rounded-lg bg-brand-50 text-brand-700 font-medium hover:bg-brand-100 disabled:opacity-50">
                  {calibrating ? 'Applying...' : 'Apply Homography Calibration'}
                </button>
              )}
            </div>
            {session.video_url ? (
              <VideoGazePlayer videoUrl={session.video_url} samples={m.gaze_samples} width={640} height={360} />
            ) : (
              <GazeOverlayFallback samples={m.gaze_samples} />
            )}
            <p className="text-xs text-slate-400">
              Gaze points synchronized to video timestamp. Upload a driving video for scene-aware analysis.
            </p>
          </div>
        </div>
      )}

      {tab === 'heatmap' && (
        <div className="space-y-6">
          <div className="glass-card p-6">
            <h3 className="font-display font-semibold text-slate-800 mb-4">Fixation Density Heatmap</h3>
            <div className="flex justify-center">
              <GazeHeatmap grid={m.heatmap_grid} width={640} height={360} />
            </div>
          </div>
          <div className="glass-card p-6">
            <h3 className="font-display font-semibold text-slate-800 mb-4">Temporal Attention Heatmap</h3>
            <TemporalHeatmap data={m.temporal_heatmap} height={180} />
          </div>
          <div className="glass-card p-6">
            <h3 className="font-display font-semibold text-slate-800 mb-4">Fixation Map</h3>
            <div className="relative bg-slate-100 border border-slate-200 rounded-xl overflow-hidden" style={{ height: 320 }}>
              <svg className="absolute inset-0 w-full h-full">
                {(m.fixations || []).slice(0, 100).map((f, i) => (
                  <circle key={i}
                    cx={`${(f.x / (m.video_width || 1920)) * 100}%`}
                    cy={`${(f.y / (m.video_height || 1080)) * 100}%`}
                    r={Math.min(20, Math.max(4, f.duration * 15))}
                    fill="rgba(12,141,232,0.18)" stroke="rgba(12,141,232,0.6)" strokeWidth={1.5} />
                ))}
              </svg>
            </div>
          </div>
        </div>
      )}

      {tab === 'timeline' && (
        <div className="glass-card p-6 space-y-4">
          <h3 className="font-display font-semibold text-slate-800">Attention Timeline</h3>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={timeline}>
              <defs>
                <linearGradient id="timelineGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#0c8de8" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#0c8de8" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="t" tickFormatter={v => `${v}s`} tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
              <Tooltip formatter={(v) => [`${v}%`, 'Road Attention']} labelFormatter={v => `Time: ${v}s`}
                contentStyle={{ background: 'white', border: '1px solid #e2e8f0', borderRadius: 12 }} />
              <ReferenceLine y={60} stroke="#f59e0b" strokeDasharray="4 3" />
              <Area type="monotone" dataKey="road_pct" stroke="#0c8de8" strokeWidth={2} fill="url(#timelineGrad)" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {tab === 'coaching' && (
        <div className="space-y-6">
          <div className="glass-card p-6">
            <h3 className="font-display font-semibold text-slate-800 mb-4">AI Coaching Report</h3>
            <div className="coaching-bubble p-5 rounded-xl">
              {session.coaching_report.split('\n').map((line, i) => (
                <p key={i} className={`text-sm ${
                  line.startsWith('**') ? 'font-semibold text-slate-800 mt-3' :
                  line.startsWith('•') ? 'text-slate-600 ml-3' :
                  line.startsWith('⚠️') ? 'text-amber-700' :
                  line.startsWith('✅') ? 'text-green-700' : 'text-slate-600'
                }`}>{line.replace(/\*\*/g, '')}</p>
              ))}
            </div>
          </div>
          {(session.coaching_insights || []).length > 0 && (
            <div className="glass-card p-6">
              <h3 className="font-display font-semibold text-slate-800 mb-4">Explainable Insights</h3>
              <div className="space-y-3">
                {session.coaching_insights.map((ins, i) => (
                  <div key={i} className={`p-4 rounded-xl border ${
                    ins.severity === 'warning' ? 'border-amber-100 bg-amber-50' :
                    ins.severity === 'success' ? 'border-green-100 bg-green-50' : 'border-slate-100 bg-slate-50'
                  }`}>
                    <p className="text-xs font-semibold text-slate-500 uppercase">{ins.metric}: {ins.value}</p>
                    <p className="text-sm text-slate-700 mt-1">{ins.evidence}</p>
                    <p className="text-sm text-brand-700 mt-2">→ {ins.recommendation}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
