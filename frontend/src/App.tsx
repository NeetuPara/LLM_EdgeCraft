import {
  BrowserRouter, Routes, Route, Navigate, useLocation
} from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { Toaster } from 'sonner'
import { useEffect, lazy, Suspense, Component, ReactNode } from 'react'
import AnimatedBackground from '@/components/AnimatedBackground'
import ProtectedRoute from '@/components/ProtectedRoute'
import { useAuthStore } from '@/stores/auth-store'

// ── Lazy-load every screen so one broken import doesn't crash the whole app ──
const LoginScreen          = lazy(() => import('@/screens/LoginScreen'))
const DashboardScreen      = lazy(() => import('@/screens/DashboardScreen'))
const ModelSelectionScreen = lazy(() => import('@/screens/wizard/ModelSelectionScreen'))
const DatasetScreen        = lazy(() => import('@/screens/wizard/DatasetScreen'))
const HyperparamsScreen    = lazy(() => import('@/screens/wizard/HyperparamsScreen'))
const TrainingSummaryScreen= lazy(() => import('@/screens/wizard/TrainingSummaryScreen'))
const TrainingScreen       = lazy(() => import('@/screens/TrainingScreen'))
const ChatScreen           = lazy(() => import('@/screens/ChatScreen'))
const ExportScreen         = lazy(() => import('@/screens/ExportScreen'))
const DataRecipesScreen    = lazy(() => import('@/screens/DataRecipesScreen'))
const RecipeEditorScreen   = lazy(() => import('@/screens/RecipeEditorScreen'))

// ── Loading spinner shown while a lazy screen loads ──
function ScreenLoader() {
  return (
    <div className="h-screen flex items-center justify-center">
      <div className="flex flex-col items-center gap-4">
        <div className="w-10 h-10 border-2 border-cap-cyan/20 border-t-cap-cyan rounded-full animate-spin" />
        <p className="text-slate-500 text-sm">Loading...</p>
      </div>
    </div>
  )
}

// ── Global error boundary — shows error instead of blank screen ──
interface EBState { hasError: boolean; error: string }
class ErrorBoundary extends Component<{ children: ReactNode }, EBState> {
  state: EBState = { hasError: false, error: '' }
  static getDerivedStateFromError(err: Error): EBState {
    return { hasError: true, error: err?.message ?? String(err) }
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="h-screen flex items-center justify-center p-8">
          <div className="glass-card max-w-lg w-full text-center">
            <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-red-500/10 border border-red-500/20 flex items-center justify-center">
              <span className="text-red-400 text-xl">!</span>
            </div>
            <h2 className="text-lg font-bold text-slate-200 mb-2">Something went wrong</h2>
            <p className="text-sm text-slate-400 font-mono bg-slate-900/50 rounded-lg p-3 mb-4 text-left break-all">
              {this.state.error}
            </p>
            <button
              onClick={() => { this.setState({ hasError: false, error: '' }); window.location.href = '/' }}
              className="btn-primary text-sm py-2"
            >
              Reload App
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

// ── Subtle slide-up transition (opacity always 1) ──
function PageTransition({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 1, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 1 }}
      transition={{ duration: 0.18, ease: 'easeOut' }}
      style={{ minHeight: '100%' }}
    >
      {children}
    </motion.div>
  )
}

function AnimatedRoutes() {
  const location = useLocation()
  const { checkAuth } = useAuthStore()

  useEffect(() => { checkAuth() }, [checkAuth])

  useEffect(() => {
    const handle = () => {
      sessionStorage.setItem('session_expired', '1')
      useAuthStore.getState().logout()
      window.location.href = '/login'
    }
    window.addEventListener('auth:session-expired', handle)
    return () => window.removeEventListener('auth:session-expired', handle)
  }, [])

  const protect = (screen: React.ReactNode) => (
    <ProtectedRoute>
      <PageTransition>
        <Suspense fallback={<ScreenLoader />}>
          {screen}
        </Suspense>
      </PageTransition>
    </ProtectedRoute>
  )

  const pub = (screen: React.ReactNode) => (
    <PageTransition>
      <Suspense fallback={<ScreenLoader />}>
        {screen}
      </Suspense>
    </PageTransition>
  )

  return (
    <AnimatePresence mode="sync" initial={false}>
      <Routes location={location} key={location.pathname}>

        {/* Public */}
        <Route path="/login"   element={pub(<LoginScreen />)} />

        {/* Dashboard */}
        <Route path="/dashboard" element={protect(<DashboardScreen />)} />

        {/* Training Wizard */}
        <Route path="/new/model"   element={protect(<ModelSelectionScreen />)} />
        <Route path="/new/dataset" element={protect(<DatasetScreen />)} />
        <Route path="/new/params"  element={protect(<HyperparamsScreen />)} />
        <Route path="/new/summary" element={protect(<TrainingSummaryScreen />)} />

        {/* Live Training */}
        <Route path="/training" element={protect(<TrainingScreen />)} />

        {/* Chat */}
        <Route path="/chat" element={protect(<ChatScreen />)} />

        {/* Export */}
        <Route path="/export" element={protect(<ExportScreen />)} />

        {/* Data Recipes */}
        <Route path="/recipes"     element={protect(<DataRecipesScreen />)} />
        <Route path="/recipes/:id" element={protect(<RecipeEditorScreen />)} />

        {/* Fallback */}
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />

      </Routes>
    </AnimatePresence>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AnimatedBackground className="fixed inset-0 w-full h-full z-0" />
      <div className="relative z-10 min-h-screen">
        <ErrorBoundary>
          <AnimatedRoutes />
        </ErrorBoundary>
      </div>
      <Toaster
        theme="dark"
        position="top-right"
        toastOptions={{
          style: {
            background: 'rgba(30, 41, 59, 0.95)',
            border: '1px solid rgba(255,255,255,0.08)',
            color: '#E2E8F0',
            backdropFilter: 'blur(12px)',
          },
        }}
      />
    </BrowserRouter>
  )
}
