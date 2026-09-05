import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'

export function PageHeader({ title, subtitle, action }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
      className="flex items-center justify-between flex-wrap gap-4"
    >
      <div>
        <h1 className="font-display font-bold text-3xl md:text-4xl text-gradient tracking-tight">{title}</h1>
        {subtitle && <p className="text-slate-600 text-sm mt-1.5 max-w-xl">{subtitle}</p>}
      </div>
      {action}
    </motion.div>
  )
}

export function PrimaryButton({ to, onClick, children, className = '' }) {
  const cls = `btn-primary inline-flex items-center gap-2 ${className}`
  if (to) return <Link to={to} className={cls}>{children}</Link>
  return <button type="button" onClick={onClick} className={cls}>{children}</button>
}

export function MotionSection({ children, className = '', delay = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className={className}
    >
      {children}
    </motion.div>
  )
}
