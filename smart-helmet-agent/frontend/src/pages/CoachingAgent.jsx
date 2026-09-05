import React, { useEffect, useState, useRef } from 'react'
import { getSessions, getLLMCoaching } from '../utils/api.js'
import { Bot, Send, Loader2, ChevronDown, Sparkles, User } from 'lucide-react'
import { RiskBadge, ScoreRing } from '../components/UI.jsx'
import { PageHeader } from '../components/layout/PageHeader.jsx'

const STARTER_QUESTIONS = [
  "What are my biggest weaknesses?",
  "How can I improve my road attention?",
  "Am I ready for highway driving?",
  "What's causing my distraction events?",
  "Give me a full coaching report",
  "How does my gaze stability compare to safe drivers?",
]

function Message({ role, content, typing, insights }) {
  return (
    <div className={`flex items-start gap-3 ${role === 'user' ? 'flex-row-reverse' : ''}`}>
      <div className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 shadow ${
        role === 'user' ? 'bg-brand-500 text-white' : 'bg-white border border-slate-200'
      }`}>
        {role === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4 text-brand-500" />}
      </div>
      <div className={`max-w-[80%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${
        role === 'user'
          ? 'bg-brand-500 text-white rounded-tr-md'
          : 'bg-white border border-slate-100 text-slate-700 rounded-tl-md shadow-sm'
      } ${typing ? 'cursor-blink' : ''}`}>
        {content.split('\n').map((line, i) => (
          <p key={i} className={`${i > 0 ? 'mt-1.5' : ''} ${line.startsWith('**') ? 'font-semibold' : ''}`}>
            {line.replace(/\*\*/g, '')}
          </p>
        ))}
        {insights?.length > 0 && (
          <div className="mt-3 pt-3 border-t border-slate-100 space-y-2">
            <p className="text-xs font-semibold text-slate-500 uppercase">Evidence</p>
            {insights.slice(0, 3).map((ins, i) => (
              <p key={i} className="text-xs text-slate-600">• {ins.evidence}</p>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default function CoachingAgent() {
  const [sessions, setSessions] = useState([])
  const [sessionId, setSessionId] = useState('')
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef()

  useEffect(() => {
    getSessions().then(s => {
      setSessions(s)
      if (s.length > 0) setSessionId(s[0].session_id)
    })
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (sessionId) {
      setMessages([{
        role: 'assistant',
        content: `Hello! I'm your Smart Helmet AI Coach. I've loaded session #${sessionId}. Ask me anything about your driving attention, or choose a starter question below.`,
      }])
    }
  }, [sessionId])

  const send = async (question) => {
    const q = question || input.trim()
    if (!q || !sessionId) return
    setInput('')
    setMessages(m => [...m, { role: 'user', content: q }])
    setLoading(true)

    try {
      const { coaching, llm, insights } = await getLLMCoaching(sessionId, q)
      setMessages(m => [...m, {
        role: 'assistant',
        content: coaching + (!llm ? '\n\n_(Using rule-based coaching — AWS Bedrock unavailable)_' : ''),
        insights: insights || [],
      }])
    } catch (e) {
      setMessages(m => [...m, {
        role: 'assistant',
        content: 'Failed to get coaching response. Make sure the backend is running.',
      }])
    } finally {
      setLoading(false)
    }
  }

  const selectedSession = sessions.find(s => s.session_id === sessionId)

  return (
    <div className="max-w-3xl mx-auto space-y-6 h-full flex flex-col">
      <PageHeader title="AI Coaching Agent" subtitle="Personalized driving attention coach with explainable insights" />

      {/* Session selector */}
      <div className="glass-card p-4 flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-brand-500" />
          <span className="text-sm font-medium text-slate-700">Session:</span>
        </div>
        <select value={sessionId} onChange={e => setSessionId(e.target.value)}
          className="flex-1 px-3 py-2 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-brand-400 bg-white">
          {sessions.map(s => (
            <option key={s.session_id} value={s.session_id}>
              #{s.session_id} — Score {s.metrics.attention_score}/100 — {s.metrics.risk_level} Risk
            </option>
          ))}
        </select>
        {selectedSession && (
          <div className="flex items-center gap-2">
            <RiskBadge level={selectedSession.metrics.risk_level} />
          </div>
        )}
      </div>

      {selectedSession && (
        <div className="glass-card p-4 flex items-center gap-6 overflow-x-auto">
          <ScoreRing score={Math.round(selectedSession.metrics.attention_score)} size={80} label="" />
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 flex-1">
            {[
              ['Road Attention', selectedSession.metrics.road_attention_pct + '%'],
              ['Stability', Math.round(selectedSession.metrics.gaze_stability) + '%'],
              ['Distractions', selectedSession.metrics.distraction_events],
              ['Level', selectedSession.metrics.driver_level],
            ].map(([k, v]) => (
              <div key={k} className="metric-pill rounded-xl p-2.5 text-center">
                <p className="text-xs text-slate-400">{k}</p>
                <p className="font-semibold text-slate-800 text-sm font-mono">{v}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Chat */}
      <div className="glass-card flex-1 flex flex-col min-h-[400px]">
        {/* Messages */}
        <div className="flex-1 p-5 space-y-4 overflow-y-auto max-h-[400px]">
          {messages.map((m, i) => (
            <Message key={i} role={m.role} content={m.content} insights={m.insights} />
          ))}
          {loading && (
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-xl bg-white border border-slate-200 flex items-center justify-center">
                <Bot className="w-4 h-4 text-brand-500" />
              </div>
              <div className="bg-white border border-slate-100 rounded-2xl rounded-tl-md px-4 py-3 shadow-sm">
                <div className="flex gap-1">
                  {[0, 1, 2].map(i => (
                    <div key={i} className="w-2 h-2 rounded-full bg-brand-300 animate-bounce"
                      style={{ animationDelay: `${i * 0.15}s` }} />
                  ))}
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Starter questions */}
        {messages.length <= 1 && (
          <div className="px-5 pb-3 flex flex-wrap gap-2">
            {STARTER_QUESTIONS.map(q => (
              <button key={q} onClick={() => send(q)}
                className="px-3 py-1.5 rounded-full bg-brand-50 border border-brand-100 text-xs text-brand-700 font-medium hover:bg-brand-100 transition-colors">
                {q}
              </button>
            ))}
          </div>
        )}

        {/* Input */}
        <div className="p-4 border-t border-slate-100 flex gap-2">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
            placeholder="Ask your AI coach anything..."
            className="flex-1 px-4 py-2.5 rounded-xl border border-slate-200 text-sm focus:outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-100 bg-white"
            disabled={loading || !sessionId}
          />
          <button onClick={() => send()}
            disabled={loading || !input.trim() || !sessionId}
            className="w-10 h-10 rounded-xl bg-brand-500 text-white flex items-center justify-center hover:bg-brand-600 transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </button>
        </div>
      </div>

      <div className="text-xs text-slate-400 text-center px-4">
        Powered by <code className="font-mono bg-slate-100 px-1 rounded">Claude on AWS Bedrock</code> for AI coaching.
      </div>
    </div>
  )
}
