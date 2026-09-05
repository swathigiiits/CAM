import React, { useState } from 'react'
import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LayoutDashboard, Upload, TrendingUp,
  GitCompare, Bot, Menu, ChevronRight
} from 'lucide-react'
import { GlobalBackground } from './components/three/GlobalBackground.jsx'
import Dashboard from './pages/Dashboard.jsx'
import UploadSession from './pages/UploadSession.jsx'
import SessionDetail from './pages/SessionDetail.jsx'
import Progress from './pages/Progress.jsx'
import Compare from './pages/Compare.jsx'
import CoachingAgent from './pages/CoachingAgent.jsx'

const NAV = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/upload', icon: Upload, label: 'Upload Session' },
  { to: '/progress', icon: TrendingUp, label: 'Progress' },
  { to: '/compare', icon: GitCompare, label: 'Compare' },
  { to: '/coach', icon: Bot, label: 'AI Coach' },
]

function Sidebar({ open, onClose }) {
  return (
    <>
      {open && (
        <div className="fixed inset-0 bg-slate-900/20 backdrop-blur-sm z-30 lg:hidden" onClick={onClose} />
      )}

      <aside className={`
        fixed top-0 left-0 h-full w-64 z-40
        glass-card-strong
        border-r border-slate-200/80
        flex flex-col
        transition-transform duration-300 ease-out
        ${open ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        rounded-none border-y-0 border-l-0
      `}>
        <div className="p-6 border-b border-slate-100">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-brand-400 to-brand-700 flex items-center justify-center glow-brand">
              <svg className="w-5 h-5 text-white" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C8.13 2 5 5.13 5 9c0 2.38 1.19 4.47 3 5.74V17c0 .55.45 1 1 1h6c.55 0 1-.45 1-1v-2.26C17.81 13.47 19 11.38 19 9c0-3.87-3.13-7-7-7zm2 14h-4v-1h4v1z" />
              </svg>
            </div>
            <div>
              <p className="font-display font-bold text-sm text-slate-900 leading-tight">Smart Helmet</p>
              <p className="text-xs text-brand-600 font-mono">Analytics Agent</p>
            </div>
          </div>
        </div>

        <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              onClick={onClose}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 group
                ${isActive
                  ? 'nav-active font-semibold'
                  : 'text-slate-600 hover:bg-brand-50 hover:text-brand-700'
                }`
              }
            >
              <Icon className="w-4 h-4 shrink-0" />
              {label}
              <ChevronRight className="w-3 h-3 ml-auto opacity-0 group-hover:opacity-100 transition-opacity" />
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-slate-100">
          <div className="metric-pill rounded-xl p-3 text-xs text-slate-500">
            <p className="font-mono font-medium text-brand-600">v2.0.0 · 3D</p>
            <p className="mt-0.5">Attention intelligence</p>
          </div>
        </div>
      </aside>
    </>
  )
}

function AnimatedRoutes() {
  const location = useLocation()
  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -6 }}
        transition={{ duration: 0.3 }}
      >
        <Routes location={location}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/upload" element={<UploadSession />} />
          <Route path="/sessions/:id" element={<SessionDetail />} />
          <Route path="/progress" element={<Progress />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/coach" element={<CoachingAgent />} />
        </Routes>
      </motion.div>
    </AnimatePresence>
  )
}

function Layout({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="flex min-h-screen relative">
      <GlobalBackground />
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      <div className="flex-1 lg:ml-64 flex flex-col min-h-screen relative z-10">
        <header className="sticky top-0 z-20 glass-card border-0 border-b border-slate-200/80 rounded-none px-6 h-16 flex items-center gap-4">
          <button
            className="lg:hidden p-2 rounded-lg hover:bg-slate-100 text-slate-600 transition-colors"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu className="w-5 h-5" />
          </button>
          <div className="flex-1" />
          <div className="flex items-center gap-2 text-xs text-slate-500 font-mono px-3 py-1.5 rounded-full border border-slate-200 bg-white/80">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            Backend Online
          </div>
        </header>

        <main className="flex-1 p-6 md:p-8 max-w-[1600px] mx-auto w-full">
          {children}
        </main>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <AnimatedRoutes />
      </Layout>
    </BrowserRouter>
  )
}
