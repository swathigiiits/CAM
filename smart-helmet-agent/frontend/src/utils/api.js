import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL || '/api'

export const api = axios.create({ baseURL: BASE })

export const uploadSession = async (formData, onProgress) => {
  const res = await api.post('/sessions/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => onProgress && onProgress(Math.round(e.loaded * 100 / e.total)),
  })
  return res.data
}

export const getSessions = async (driverId) => {
  const res = await api.get('/sessions', { params: driverId ? { driver_id: driverId } : {} })
  return res.data
}

export const getSession = async (id) => {
  const res = await api.get(`/sessions/${id}`)
  return res.data
}

export const getProgress = async (driverId) => {
  const res = await api.get(`/drivers/${driverId}/progress`)
  return res.data
}

export const compareSessions = async (ids) => {
  const res = await api.post('/sessions/compare', { session_ids: ids })
  return res.data
}

export const getLLMCoaching = async (sessionId, question) => {
  const res = await api.post(`/sessions/${sessionId}/coaching/llm`, { question })
  return res.data
}

export const setCalibration = async (sessionId, srcPoints, dstPoints) => {
  const res = await api.post(`/sessions/${sessionId}/calibration`, {
    src_points: srcPoints,
    dst_points: dstPoints,
  })
  return res.data
}

export const getSessionVideoUrl = (session) => session?.video_url || null

export const generateDemoCSV = async () => {
  const res = await api.get('/demo/generate-csv')
  return res.data
}
