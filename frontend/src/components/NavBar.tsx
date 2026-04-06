import { useState, useEffect, useRef } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import {
  LayoutDashboard, MessageSquare, Download, FlaskConical,
  LogOut, User, CheckCircle, ChevronDown, Cpu,
} from 'lucide-react'
import Logo from './Logo'
import { useAuthStore } from '@/stores/auth-store'
import { PIPELINE_STAGES } from '@/config/constants'

const NAV_LINKS = [
  { label: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
  { label: 'Chat',      path: '/chat',      icon: MessageSquare },
  { label: 'Export',    path: '/export',    icon: Download },
  // { label: 'Recipes', path: '/recipes', icon: FlaskConical },  // hidden until backend implemented
]

// Wizard screens that show the pipeline progress pills
const WIZARD_PATHS = ['/new/model', '/new/dataset', '/new/params', '/new/summary', '/training']

export default function NavBar() {
  const navigate = useNavigate()
  const location = useLocation()
  const { user, logout } = useAuthStore()

  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false)
  const [showProfileDropdown, setShowProfileDropdown] = useState(false)
  const profileDropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!showProfileDropdown) return
    const close = (e: MouseEvent) => {
      if (profileDropdownRef.current && !profileDropdownRef.current.contains(e.target as Node)) {
        setShowProfileDropdown(false)
      }
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [showProfileDropdown])

  const currentPath = location.pathname
  const showPills = WIZARD_PATHS.some(p => currentPath.startsWith(p))

  const currentStageIndex = PIPELINE_STAGES.findIndex(s =>
    s.screens.some(sc => currentPath.startsWith(sc))
  )

  const handleLogout = () => {
    setShowLogoutConfirm(false)
    logout()
    navigate('/login')
  }

  return (
    <>
      <nav className="sticky top-0 z-50 px-6 py-3">
        <div className="max-w-7xl mx-auto w-full flex flex-col gap-2">

          {/* ── Main Island ── */}
          <div
            className="bg-slate-900/50 backdrop-blur-[20px] border border-white/[0.08] rounded-xl px-4 py-2.5"
            style={{ transform: 'translateZ(0)', willChange: 'transform' }}
          >
            <div className="flex items-center justify-between">

              {/* Left: Logo */}
              <button
                onClick={() => navigate('/dashboard')}
                className="flex items-center gap-2 hover:opacity-80 transition-opacity"
              >
                <Logo variant="full" size="md" />
              </button>

              {/* Right: Nav links + profile */}
              <div className="flex items-center gap-1">

                {/* Nav links */}
                {NAV_LINKS.map(({ label, path, icon: Icon }) => {
                  const active = currentPath === path || currentPath.startsWith(path + '/')
                  return (
                    <button
                      key={path}
                      onClick={() => navigate(path)}
                      className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                        active
                          ? 'bg-cap-cyan/10 text-cap-cyan border border-cap-cyan/20'
                          : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
                      }`}
                    >
                      <Icon size={15} />
                      {label}
                    </button>
                  )
                })}

                <div className="w-px h-5 bg-white/10 mx-2" />

                {/* GPU indicator */}
                <div className="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 text-xs text-slate-500">
                  <Cpu size={12} />
                  <span>GPU</span>
                </div>

                {/* Profile dropdown */}
                <div className="relative ml-1" ref={profileDropdownRef}>
                  <button
                    onClick={() => setShowProfileDropdown(!showProfileDropdown)}
                    className="flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-white/5 transition-colors"
                  >
                    <div className="w-8 h-8 rounded-full bg-cap-cyan flex items-center justify-center text-white text-sm font-semibold shadow-lg shadow-cap-cyan/20">
                      {(user?.name || user?.email)?.charAt(0).toUpperCase() ?? <User size={14} />}
                    </div>
                    <ChevronDown size={14} className="text-slate-500" />
                  </button>

                  {showProfileDropdown && (
                    <div className="absolute right-0 mt-2 w-48 bg-slate-900/95 backdrop-blur-xl border border-slate-700 rounded-xl shadow-2xl z-50 py-1">
                      <div className="px-4 py-2.5 border-b border-slate-800">
                        <p className="text-sm font-medium text-slate-200">{user?.name || user?.email}</p>
                        <p className="text-xs text-slate-500">Unsloth Studio</p>
                      </div>
                      <button
                        onClick={() => { setShowLogoutConfirm(true); setShowProfileDropdown(false) }}
                        className="w-full flex items-center gap-2.5 px-4 py-2.5 text-sm text-red-400 hover:bg-red-500/10 transition-colors"
                      >
                        <LogOut size={14} />
                        Sign Out
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* ── Pipeline Progress Pills ── */}
          {showPills && (
            <div
              className="bg-slate-900/35 backdrop-blur-[16px] border border-white/[0.06] rounded-[10px] px-4 py-2.5 flex items-center justify-between gap-3 flex-wrap"
              style={{ transform: 'translateZ(0)', willChange: 'transform' }}
            >
              {PIPELINE_STAGES.map((stage, idx) => {
                const isCurrent = idx === currentStageIndex
                const isCompleted = idx < currentStageIndex

                return (
                  <button
                    key={stage.id}
                    onClick={() => isCompleted && navigate(stage.screens[0])}
                    className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${
                      isCompleted
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/15 cursor-pointer'
                        : isCurrent
                          ? 'bg-cap-cyan/10 text-cap-cyan border border-cap-cyan/30 ring-1 ring-cap-cyan/20'
                          : 'bg-slate-800/50 text-slate-500 border border-white/[0.04]'
                    }`}
                  >
                    {isCompleted ? (
                      <CheckCircle size={13} />
                    ) : (
                      <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${
                        isCurrent ? 'bg-cap-cyan text-white' : 'bg-slate-700 text-slate-500'
                      }`}>
                        {idx + 1}
                      </span>
                    )}
                    {stage.shortLabel}
                  </button>
                )
              })}
            </div>
          )}

        </div>
      </nav>

      {/* Logout Confirmation Modal */}
      {showLogoutConfirm && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm"
          onClick={() => setShowLogoutConfirm(false)}
        >
          <div
            className="bg-slate-900/80 backdrop-blur-xl border border-white/[0.08] rounded-2xl shadow-2xl p-6 w-80 text-center"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-red-500/10 border border-red-500/20 flex items-center justify-center">
              <LogOut className="text-red-400" size={22} />
            </div>
            <h3 className="text-white text-lg font-semibold mb-1">Sign Out</h3>
            <p className="text-slate-400 text-sm mb-6">Are you sure you want to sign out?</p>
            <div className="flex gap-3">
              <button
                onClick={() => setShowLogoutConfirm(false)}
                className="flex-1 px-4 py-2.5 rounded-xl bg-white/[0.05] border border-white/[0.08] text-slate-300 hover:bg-white/[0.1] text-sm font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleLogout}
                className="flex-1 px-4 py-2.5 rounded-xl bg-red-500/20 border border-red-500/30 text-red-400 hover:bg-red-500/30 text-sm font-medium transition-colors"
              >
                Sign Out
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
