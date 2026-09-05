import React, { useRef, useEffect, useState, useCallback } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || ''

export function VideoGazePlayer({
  videoUrl,
  samples = [],
  width = 640,
  height = 360,
  showTrail = true,
}) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [currentT, setCurrentT] = useState(0)

  const resolvedUrl = videoUrl?.startsWith('http')
    ? videoUrl
    : `${API_BASE.replace(/\/api$/, '')}${videoUrl}`

  const drawOverlay = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const t = video.currentTime
    setCurrentT(t)

    const window_s = showTrail ? 1.5 : 0.05
    const visible = samples.filter(s => s.t >= t - window_s && s.t <= t + 0.05)

    visible.forEach((pt, i) => {
      const age = t - pt.t
      const alpha = showTrail ? Math.max(0.15, 1 - age / window_s) : 0.9
      const x = pt.x * canvas.width
      const y = pt.y * canvas.height
      const r = 4 + pt.p * 6

      ctx.beginPath()
      ctx.arc(x, y, r, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(12, 141, 232, ${alpha * 0.5})`
      ctx.fill()
      ctx.strokeStyle = `rgba(12, 141, 232, ${alpha})`
      ctx.lineWidth = 1.5
      ctx.stroke()

      if (i === visible.length - 1) {
        ctx.beginPath()
        ctx.arc(x, y, r + 4, 0, Math.PI * 2)
        ctx.strokeStyle = 'rgba(255,255,255,0.8)'
        ctx.lineWidth = 2
        ctx.stroke()
      }
    })
  }, [samples, showTrail])

  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const onTime = () => drawOverlay()
    video.addEventListener('timeupdate', onTime)
    return () => video.removeEventListener('timeupdate', onTime)
  }, [drawOverlay])

  if (!videoUrl) {
    return (
      <div className="flex items-center justify-center bg-slate-800 rounded-xl text-slate-400 text-sm"
        style={{ width, height }}>
        No video uploaded — showing gaze overlay only
      </div>
    )
  }

  return (
    <div className="relative rounded-xl overflow-hidden bg-black" style={{ width, maxWidth: '100%' }}>
      <video
        ref={videoRef}
        src={resolvedUrl}
        className="w-full block"
        style={{ height }}
        controls
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        crossOrigin="anonymous"
      />
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />
      <div className="absolute top-2 left-2 text-xs font-mono bg-white/90 text-slate-800 px-2 py-1 rounded-lg border border-slate-200 shadow-sm">
        {playing ? '▶' : '⏸'} {currentT.toFixed(2)}s · Synced Gaze
      </div>
    </div>
  )
}

export function TemporalHeatmap({ data, height = 200 }) {
  if (!data || !data.length) return null
  const cols = data[0]?.length || 10

  return (
    <div className="space-y-2">
      <div className="flex gap-0.5 rounded-xl overflow-hidden border border-slate-100" style={{ height }}>
        {data.map((row, ri) => (
          <div key={ri} className="flex-1 flex flex-col-reverse gap-0">
            {row.map((val, ci) => {
              const intensity = val / 100
              const r = Math.round(255 * intensity)
              const g = Math.round(255 * (1 - intensity * 0.5))
              return (
                <div
                  key={ci}
                  className="flex-1"
                  style={{ background: `rgba(${r},${g},0,${0.3 + intensity * 0.6})` }}
                  title={`T${ri} zone${ci}: ${val}%`}
                />
              )
            })}
          </div>
        ))}
      </div>
      <div className="flex justify-between text-xs text-slate-400">
        <span>Session start</span>
        <span>Time →</span>
        <span>Session end</span>
      </div>
      <p className="text-xs text-slate-400 text-center">
        Temporal attention heatmap — vertical = screen position, horizontal = time
      </p>
    </div>
  )
}
