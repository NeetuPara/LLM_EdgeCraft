import { useState, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { Mail, Lock, ArrowRight, AlertCircle, Clock, User, Zap, UserPlus } from 'lucide-react'
import { motion } from 'framer-motion'
import { useAuthStore } from '@/stores/auth-store'
import Logo from '@/components/Logo'
import { isMockMode } from '@/api/mock'

const IS_DEMO = isMockMode()

type AuthTab = 'login' | 'signup'

export default function LoginScreen() {
  const navigate = useNavigate()
  const location = useLocation()
  const { login, signup, isAuthenticated } = useAuthStore()

  const [tab, setTab] = useState<AuthTab>('login')
  const [email, setEmail] = useState(IS_DEMO ? 'demo@example.com' : '')
  const [password, setPassword] = useState(IS_DEMO ? 'demo' : '')
  const [name, setName] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [sessionExpired, setSessionExpired] = useState(false)

  useEffect(() => {
    if (sessionStorage.getItem('session_expired')) {
      setSessionExpired(true)
      sessionStorage.removeItem('session_expired')
    }
  }, [])

  useEffect(() => {
    if (isAuthenticated) {
      const from = (location.state as { from?: Location })?.from?.pathname ?? '/dashboard'
      navigate(from, { replace: true })
    }
  }, [isAuthenticated, navigate, location])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      if (tab === 'login') {
        await login(email, password)
      } else {
        if (!name.trim()) { setError('Please enter your name.'); setLoading(false); return }
        await signup(email, password, name)
      }
      navigate('/dashboard', { replace: true })
    } catch (err: unknown) {
      const msg = (err as { message?: string })?.message ?? 'Something went wrong.'
      setError(tab === 'login' ? 'Invalid email or password.' : msg)
    } finally {
      setLoading(false)
    }
  }

  const handleDemoLogin = async () => {
    setLoading(true)
    try {
      await login('demo@example.com', 'demo')
      navigate('/dashboard', { replace: true })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <motion.div
        initial={{ opacity: 1, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: 'easeOut' }}
        className="w-full max-w-md"
      >
        {/* Logo */}
        <div className="text-center mb-8">
          <Logo variant="full" size="lg" className="justify-center" />
          <p className="text-slate-400 mt-3 text-sm">LLM Fine-tuning Studio</p>
        </div>

        {/* Demo banner */}
        {IS_DEMO && (
          <motion.div
            initial={{ opacity: 1, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="mb-4 p-4 bg-cap-cyan/10 border border-cap-cyan/25 rounded-xl flex items-center justify-between gap-3"
          >
            <div className="flex items-center gap-2 text-sm text-cap-cyan">
              <Zap size={16} className="shrink-0" />
              <span><span className="font-semibold">Demo Mode</span> — no backend needed</span>
            </div>
            <button
              onClick={handleDemoLogin}
              disabled={loading}
              className="shrink-0 flex items-center gap-1.5 px-3 py-1.5 bg-cap-cyan text-white text-xs font-bold rounded-lg hover:bg-cap-cyan-dark transition-colors disabled:opacity-50"
            >
              <Zap size={12} /> Enter Demo
            </button>
          </motion.div>
        )}

        {/* Card */}
        <div className="bg-slate-800/40 backdrop-blur-xl border border-white/10 rounded-2xl p-8 shadow-2xl shadow-black/30">

          {/* Tabs */}
          <div className="flex rounded-xl bg-slate-900/50 p-1 mb-6">
            {(['login', 'signup'] as AuthTab[]).map(t => (
              <button
                key={t}
                onClick={() => { setTab(t); setError('') }}
                className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-sm font-medium transition-all ${
                  tab === t
                    ? 'bg-cap-blue/80 text-white shadow'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {t === 'login' ? <><Lock size={13} /> Sign In</> : <><UserPlus size={13} /> Create Account</>}
              </button>
            ))}
          </div>

          {/* Banners */}
          {sessionExpired && !error && (
            <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-4 mb-5 flex items-start gap-3">
              <Clock className="text-amber-400 shrink-0 mt-0.5" size={18} />
              <p className="text-sm text-amber-200">Your session expired. Please sign in again.</p>
            </div>
          )}
          {error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 mb-5 flex items-start gap-3">
              <AlertCircle className="text-red-500 shrink-0 mt-0.5" size={18} />
              <p className="text-sm text-red-200">{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Name (signup only) */}
            {tab === 'signup' && (
              <div>
                <label className="block text-sm font-medium text-slate-400 mb-2">Full Name</label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                  <input
                    type="text"
                    value={name}
                    onChange={e => setName(e.target.value)}
                    className="glass-input pl-10"
                    placeholder="Your Name"
                    autoFocus
                  />
                </div>
              </div>
            )}

            {/* Email */}
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-2">Email Address</label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                <input
                  type="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  className="glass-input pl-10"
                  placeholder="you@example.com"
                  required
                  autoFocus={tab === 'login' && !IS_DEMO}
                  autoComplete="email"
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-2">Password</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                <input
                  type="password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  className="glass-input pl-10"
                  placeholder="••••••••"
                  required
                  autoComplete={tab === 'login' ? 'current-password' : 'new-password'}
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-[#0070AD] hover:bg-[#0088CC] text-white font-bold py-3 rounded-xl
                         transition-all transform hover:scale-[1.02] active:scale-[0.98]
                         disabled:opacity-50 disabled:cursor-not-allowed
                         shadow-lg shadow-[#0070AD]/20 flex items-center justify-center gap-2 mt-2"
            >
              {loading ? (
                <><div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Loading...</>
              ) : tab === 'login' ? (
                <>Sign In <ArrowRight size={18} /></>
              ) : (
                <>Create Account <ArrowRight size={18} /></>
              )}
            </button>
          </form>

          {!IS_DEMO && (
            <p className="text-center text-xs text-slate-600 mt-6">
              Make sure the EdgeCraft backend is running on port 8001.
            </p>
          )}
        </div>

        <p className="text-center text-xs text-slate-700 mt-6">
          Powered by <a href="https://unsloth.ai" target="_blank" rel="noopener noreferrer" className="text-slate-600 hover:text-slate-500 transition-colors">Unsloth AI</a>
        </p>
      </motion.div>
    </div>
  )
}
