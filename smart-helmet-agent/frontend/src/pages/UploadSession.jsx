import React, { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, FileText, Video, CheckCircle2, Loader2, Download, AlertCircle, X } from 'lucide-react'
import { uploadSession, generateDemoCSV } from '../utils/api.js'
import { PageHeader } from '../components/layout/PageHeader.jsx'

export default function UploadSession() {
  const navigate = useNavigate()
  const csvRef = useRef()
  const videoRef = useRef()

  const [csvFile, setCsvFile] = useState(null)
  const [videoFile, setVideoFile] = useState(null)
  const [driverId, setDriverId] = useState('')
  const [sessionType, setSessionType] = useState('city')
  const [weather, setWeather] = useState('clear')
  const [speedKmh, setSpeedKmh] = useState('')
  const [gpsLat, setGpsLat] = useState('')
  const [gpsLng, setGpsLng] = useState('')
  const [notes, setNotes] = useState('')
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState('')
  const [csvDrag, setCsvDrag] = useState(false)

  const handleCSVDrop = (e) => {
    e.preventDefault()
    setCsvDrag(false)
    const f = e.dataTransfer?.files[0] || e.target.files?.[0]
    if (f) setCsvFile(f)
  }

  const handleVideoDrop = (e) => {
    e.preventDefault()
    const f = e.dataTransfer?.files[0] || e.target.files?.[0]
    if (f) setVideoFile(f)
  }

  const handleDemoCSV = async () => {
    try {
      const { csv, rows } = await generateDemoCSV()
      const blob = new Blob([csv], { type: 'text/csv' })
      const file = new File([blob], `demo_gaze_${Date.now()}.csv`, { type: 'text/csv' })
      setCsvFile(file)
    } catch (e) {
      setError('Failed to generate demo CSV')
    }
  }

  const handleSubmit = async () => {
    if (!csvFile) { setError('Please upload a gaze CSV file'); return }
    setError('')
    setUploading(true)
    setProgress(0)

    const fd = new FormData()
    fd.append('gaze_csv', csvFile)
    if (videoFile) fd.append('video', videoFile)
    fd.append('driver_id', driverId || 'anonymous')
    fd.append('session_type', sessionType)
    fd.append('weather', weather)
    fd.append('notes', notes)
    if (speedKmh) fd.append('speed_kmh', speedKmh)
    if (gpsLat) fd.append('gps_lat', gpsLat)
    if (gpsLng) fd.append('gps_lng', gpsLng)

    try {
      const result = await uploadSession(fd, setProgress)
      navigate(`/sessions/${result.session_id}`)
    } catch (e) {
      setError(e.response?.data?.detail || 'Upload failed. Is the backend running?')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <PageHeader
        title="Upload Session"
        subtitle="Gaze CSV + optional driving video for scene-aware 3D analytics"
      />

      {error && (
        <div className="flex items-center gap-3 p-4 rounded-xl bg-red-50 border border-red-100 text-red-700 text-sm">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
          <button onClick={() => setError('')} className="ml-auto"><X className="w-4 h-4" /></button>
        </div>
      )}

      {/* CSV Upload */}
      <div className="glass-card p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="font-display font-semibold text-slate-800">Gaze CSV <span className="text-red-400">*</span></h2>
          <button onClick={handleDemoCSV}
            className="flex items-center gap-1.5 text-xs text-brand-600 font-medium hover:text-brand-800 transition-colors">
            <Download className="w-3.5 h-3.5" /> Use Demo CSV
          </button>
        </div>

        <div
          className={`drop-zone rounded-xl p-8 text-center cursor-pointer ${csvDrag ? 'drag-over' : ''} ${csvFile ? 'border-green-400 bg-green-50' : ''}`}
          onClick={() => csvRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setCsvDrag(true) }}
          onDragLeave={() => setCsvDrag(false)}
          onDrop={handleCSVDrop}
        >
          {csvFile ? (
            <div className="flex flex-col items-center gap-2">
              <CheckCircle2 className="w-10 h-10 text-green-500" />
              <p className="font-semibold text-green-700">{csvFile.name}</p>
              <p className="text-xs text-green-600">{(csvFile.size / 1024).toFixed(1)} KB</p>
              <button onClick={(e) => { e.stopPropagation(); setCsvFile(null) }}
                className="text-xs text-slate-400 hover:text-red-500 mt-1">Remove</button>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2 text-slate-400">
              <FileText className="w-10 h-10 text-brand-300" />
              <p className="font-medium text-slate-600">Drop CSV or click to browse</p>
              <p className="text-xs">Required columns: timestamp, gaze_x, gaze_y, pupil_size</p>
            </div>
          )}
          <input ref={csvRef} type="file" accept=".csv" className="hidden" onChange={handleCSVDrop} />
        </div>

        <div className="metric-pill rounded-xl p-3 text-xs font-mono text-slate-500 space-y-0.5">
          <p className="font-semibold text-slate-700">Expected format:</p>
          <p>timestamp,gaze_x,gaze_y,pupil_size</p>
          <p>0.01,512,302,0.82</p>
          <p>0.03,510,299,0.81</p>
        </div>
      </div>

      {/* Video Upload */}
      <div className="glass-card p-6 space-y-4">
        <h2 className="font-display font-semibold text-slate-800">
          Driving Video <span className="text-xs text-slate-400 font-normal">(recommended for scene AI)</span>
        </h2>
        <div
          className={`drop-zone rounded-xl p-8 text-center cursor-pointer ${videoFile ? 'border-brand-400 bg-brand-50' : ''}`}
          onClick={() => videoRef.current.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleVideoDrop}
        >
          {videoFile ? (
            <div className="flex flex-col items-center gap-2">
              <CheckCircle2 className="w-8 h-8 text-brand-500" />
              <p className="font-semibold text-brand-700">{videoFile.name}</p>
              <p className="text-xs text-brand-500">{(videoFile.size / 1024 / 1024).toFixed(1)} MB</p>
              <button onClick={(e) => { e.stopPropagation(); setVideoFile(null) }}
                className="text-xs text-slate-400 hover:text-red-500 mt-1">Remove</button>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2 text-slate-400">
              <Video className="w-10 h-10 text-brand-200" />
              <p className="font-medium text-slate-600">Drop video or click to browse</p>
              <p className="text-xs">.mp4, .mov — synced with gaze for scene analysis</p>
            </div>
          )}
          <input ref={videoRef} type="file" accept="video/*" className="hidden" onChange={handleVideoDrop} />
        </div>
      </div>

      {/* Session Metadata */}
      <div className="glass-card p-6 space-y-4">
        <h2 className="font-display font-semibold text-slate-800">Session Details</h2>
        <div className="grid grid-cols-2 gap-4">
          <div className="col-span-2">
            <label className="text-xs font-medium text-slate-500 mb-1.5 block">Driver ID</label>
            <input
              value={driverId}
              onChange={e => setDriverId(e.target.value)}
              placeholder="e.g. driver-001"
              className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100 bg-white"
            />
          </div>
          <div>
            <label className="text-xs font-medium text-slate-500 mb-1.5 block">Session Type</label>
            <select value={sessionType} onChange={e => setSessionType(e.target.value)}
              className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-brand-400 bg-white">
              <option value="city">City</option>
              <option value="highway">Highway</option>
              <option value="rural">Rural</option>
              <option value="parking">Parking</option>
            </select>
          </div>
          <div>
            <label className="text-xs font-medium text-slate-500 mb-1.5 block">Weather</label>
            <select value={weather} onChange={e => setWeather(e.target.value)}
              className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-brand-400 bg-white">
              <option value="clear">Clear</option>
              <option value="rain">Rain</option>
              <option value="fog">Fog</option>
              <option value="night">Night</option>
              <option value="snow">Snow</option>
            </select>
          </div>
          <div className="col-span-2">
            <label className="text-xs font-medium text-slate-500 mb-1.5 block">Notes</label>
            <textarea value={notes} onChange={e => setNotes(e.target.value)}
              rows={2} placeholder="Optional notes about this session..."
              className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100 bg-white resize-none" />
          </div>
          <div>
            <label className="text-xs font-medium text-slate-500 mb-1.5 block">Avg Speed (km/h)</label>
            <input type="number" value={speedKmh} onChange={e => setSpeedKmh(e.target.value)}
              placeholder="e.g. 45" min="0"
              className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-brand-400 bg-white" />
          </div>
          <div>
            <label className="text-xs font-medium text-slate-500 mb-1.5 block">GPS Latitude</label>
            <input type="number" step="any" value={gpsLat} onChange={e => setGpsLat(e.target.value)}
              placeholder="e.g. 40.7128"
              className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-brand-400 bg-white" />
          </div>
          <div>
            <label className="text-xs font-medium text-slate-500 mb-1.5 block">GPS Longitude</label>
            <input type="number" step="any" value={gpsLng} onChange={e => setGpsLng(e.target.value)}
              placeholder="e.g. -74.0060"
              className="w-full px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-brand-400 bg-white" />
          </div>
        </div>
      </div>

      {/* Progress */}
      {uploading && (
        <div className="glass-card p-4 space-y-2">
          <div className="flex items-center gap-2 text-sm text-brand-600">
            <Loader2 className="w-4 h-4 animate-spin" />
            Processing gaze data... {progress}%
          </div>
          <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
            <div className="progress-fill h-full" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      <button
        onClick={handleSubmit}
        disabled={uploading || !csvFile}
        className="w-full py-3.5 rounded-xl bg-gradient-to-r from-brand-600 to-brand-400 text-white font-semibold text-sm shadow-lg hover:shadow-xl hover:scale-[1.01] transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
      >
        {uploading ? (
          <><Loader2 className="w-4 h-4 animate-spin" /> Analyzing Session...</>
        ) : (
          <><Upload className="w-4 h-4" /> Analyze Session</>
        )}
      </button>
    </div>
  )
}
