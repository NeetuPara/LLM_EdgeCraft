import { useState, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Sparkles, Cpu, Database, ChevronRight, Trash2,
  MessageSquare, Download, TrendingDown, Clock,
  Activity, FlaskConical, Plus,
} from 'lucide-react'
import { motion } from 'framer-motion'
import NavBar from '@/components/NavBar'
import { useAuthStore } from '@/stores/auth-store'
import { trainingApi, systemApi } from '@/api/training-api'
import type { TrainingRun, HardwareInfo } from '@/types'
import { STATUS_COLORS } from '@/config/constants'
import { cn } from '@/utils/cn'

// ── Loss sparkline (inline Recharts-free mini chart) ──
function LossSparkline({ data }: { data: number[] }) {
  if (!data || data.length < 2) return null
  const w = 80, h = 24
  const min = Math.min(...data), max = Math.max(...data)
  const range = max - min || 1
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w
    const y = h - ((v - min) / range) * h
    return `${x},${y}`
  }).join(' ')

  return (
    <svg width={w} height={h} className="opacity-60">
      <polyline points={pts} fill="none" stroke="#00A5D9" strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  )
}

// ── Status Badge ──
function StatusBadge({ status }: { status: string }) {
  const colors = STATUS_COLORS[status] ?? STATUS_COLORS.cancelled
  return (
    <span className={cn('inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full border', colors.bg, colors.text, colors.border)}>
      <span className={cn('w-1.5 h-1.5 rounded-full', colors.dot, status === 'running' && 'animate-pulse')} />
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  )
}

// ── Duration formatter ──
function formatDuration(seconds?: number): string {
  if (!seconds) return '—'
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  return `${(seconds / 3600).toFixed(1)}h`
}

// ── Date formatter ──
function formatDate(iso?: string): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}

// ── Stat Card ──
function StatCard({
  label, value, icon: Icon, accent = 'cyan',
}: {
  label: string; value: string | number; icon: React.ElementType; accent?: 'cyan' | 'green' | 'amber'
}) {
  const accentMap = {
    cyan:  { text: 'text-cap-cyan',    bg: 'bg-cap-cyan/10',   border: 'border-cap-cyan/20' },
    green: { text: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20' },
    amber: { text: 'text-amber-400',   bg: 'bg-amber-500/10',   border: 'border-amber-500/20' },
  }
  const a = accentMap[accent]
  return (
    <div className="glass-card flex items-center gap-4">
      <div className={cn('p-3 rounded-xl border', a.bg, a.border)}>
        <Icon className={a.text} size={20} />
      </div>
      <div>
        <p className="text-2xl font-bold text-slate-100">{value}</p>
        <p className="text-sm text-slate-400 mt-0.5">{label}</p>
      </div>
    </div>
  )
}

// ── GPU Info Bar ──
function GpuBar({ hw }: { hw: HardwareInfo | null }) {
  if (!hw?.gpu_name) return null
  const usedGb = hw.gpu_memory_used_gb ?? 0
  const totalGb = hw.gpu_memory_total_gb ?? 1
  const pct = Math.min((usedGb / totalGb) * 100, 100)
  const util = hw.gpu_utilization ?? 0

  return (
    <div className="glass-card flex items-center gap-6 mb-6">
      <div className="flex items-center gap-2 shrink-0">
        <Cpu size={16} className="text-cap-cyan" />
        <span className="text-sm font-medium text-slate-200">{hw.gpu_name}</span>
      </div>
      <div className="flex-1 flex items-center gap-3">
        <span className="text-xs text-slate-500 shrink-0">VRAM</span>
        <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${pct}%`,
              background: pct > 80 ? '#EF4444' : pct > 60 ? '#F59E0B' : '#00A5D9',
            }}
          />
        </div>
        <span className="text-xs text-slate-400 shrink-0">
          {(usedGb ?? 0).toFixed(1)} / {(totalGb ?? 1).toFixed(1)} GB
        </span>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <Activity size={14} className="text-slate-500" />
        <span className="text-xs text-slate-400">{util}%</span>
      </div>
    </div>
  )
}

export default function DashboardScreen() {
  const navigate = useNavigate()
  const { user } = useAuthStore()

  const [runs, setRuns] = useState<TrainingRun[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [hwInfo, setHwInfo] = useState<HardwareInfo | null>(null)
  const [deleteConfirm, setDeleteConfirm] = useState<TrainingRun | null>(null)
  const [deleting, setDeleting] = useState(false)

  const fetchRuns = useCallback(async () => {
    try {
      const data = await trainingApi.listRuns()
      setRuns(data.runs)
      setTotal(data.total)
    } catch {
      // Backend not running yet — show empty state
    } finally {
      setLoading(false)
    }
  }, [])

  const fetchHw = useCallback(async () => {
    try {
      const hw = await systemApi.hardware()
      setHwInfo(hw)
    } catch {
      // ignore
    }
  }, [])

  useEffect(() => {
    fetchRuns()
    fetchHw()
  }, [fetchRuns, fetchHw])

  const handleDeleteRun = async (id: string) => {
    setDeleting(true)
    try {
      await trainingApi.deleteRun(id)
      await fetchRuns()
      setDeleteConfirm(null)
    } catch {
      // ignore
    } finally {
      setDeleting(false)
    }
  }

  const completedRuns = runs.filter(r => r.status === 'completed').length
  const activeRuns = runs.filter(r => r.status === 'running').length

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <NavBar />

      <div className="flex-1 overflow-y-auto min-h-0">
        <div className="max-w-7xl mx-auto px-6 py-8">

          {/* Header */}
          <motion.div
            initial={{ opacity: 1, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
            className="flex justify-between items-start mb-8 gap-4"
          >
            <div>
              <h1 className="text-3xl font-bold text-slate-100 font-display">
                Welcome back{user?.name ? `, ${user.name}` : user?.email ? `, ${user.email.split('@')[0]}` : ''}
              </h1>
              <p className="text-slate-400 mt-1">Manage your LLM fine-tuning runs</p>
            </div>
            <button
              onClick={() => navigate('/new/model')}
              className="btn-primary flex items-center gap-2 whitespace-nowrap"
            >
              <Plus size={18} />
              New Fine-tune
            </button>
          </motion.div>

          {/* GPU Bar */}
          <GpuBar hw={hwInfo} />

          {/* Stats Row */}
          <motion.div
            initial={{ opacity: 1, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: 0.05 }}
            className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8"
          >
            <StatCard label="Total Runs"    value={total}         icon={Sparkles} accent="cyan"  />
            <StatCard label="Completed"     value={completedRuns} icon={TrendingDown} accent="green" />
            <StatCard label="Active Now"    value={activeRuns}    icon={Activity}  accent="amber" />
          </motion.div>

          {/* Quick Actions */}
          <motion.div
            initial={{ opacity: 1, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: 0.1 }}
            className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8"
          >
            {[
              { label: 'Chat & Inference', desc: 'Test any model or fine-tuned LoRA', icon: MessageSquare, path: '/chat',   color: 'cap-cyan' },
              { label: 'Export Models',    desc: 'Export as GGUF, merged, or LoRA',  icon: Download,      path: '/export', color: 'emerald-400' },
              // Data Recipes hidden until backend is implemented
              // { label: 'Data Recipes', desc: 'Synthesize datasets visually', icon: FlaskConical, path: '/recipes', color: 'indigo-400' },
            ].map(({ label, desc, icon: Icon, path, color }) => (
              <button
                key={path}
                onClick={() => navigate(path)}
                className="glass-card-interactive flex items-center gap-4 text-left"
              >
                <div className={`p-3 rounded-xl bg-${color}/10 border border-${color}/20 shrink-0`}>
                  <Icon className={`text-${color}`} size={20} />
                </div>
                <div>
                  <p className="font-semibold text-slate-200 text-sm">{label}</p>
                  <p className="text-xs text-slate-500 mt-0.5">{desc}</p>
                </div>
                <ChevronRight size={16} className="text-slate-600 ml-auto shrink-0" />
              </button>
            ))}
          </motion.div>

          {/* Training Runs List */}
          <motion.div
            initial={{ opacity: 1, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: 0.15 }}
            className="glass-card p-0 overflow-hidden"
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-white/[0.06]">
              <h2 className="text-lg font-semibold text-slate-200 font-display">Training Runs</h2>
              <span className="text-xs text-slate-500">{total} total</span>
            </div>

            {loading ? (
              <div className="p-6 space-y-4">
                {[1, 2, 3].map(i => (
                  <div key={i} className="flex items-center gap-4 p-4 rounded-xl bg-slate-800/30 border border-white/[0.04]">
                    <div className="skeleton w-3 h-3 rounded-full" />
                    <div className="flex-1 space-y-2">
                      <div className="skeleton h-4 w-48 rounded" />
                      <div className="skeleton h-3 w-64 rounded" />
                    </div>
                    <div className="skeleton h-6 w-20 rounded-full" />
                  </div>
                ))}
              </div>
            ) : runs.length === 0 ? (
              <div className="text-center py-16 px-6">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-white/5 flex items-center justify-center">
                  <Sparkles className="text-slate-600" size={28} />
                </div>
                <p className="text-slate-400 text-lg font-medium mb-2">No training runs yet</p>
                <p className="text-slate-600 text-sm mb-6">
                  Start your first fine-tuning run to see it here.
                </p>
                <button
                  onClick={() => navigate('/new/model')}
                  className="btn-primary inline-flex items-center gap-2"
                >
                  <Plus size={16} />
                  Start Fine-tuning
                </button>
              </div>
            ) : (
              <div className="divide-y divide-white/[0.04]">
                {runs.map((run) => (
                  <div
                    key={run.id}
                    onClick={() => {
                      if (run.status === 'running') navigate('/training')
                      else if (run.status === 'completed') navigate(`/run/${run.id}`)
                    }}
                    className={cn(
                      'flex items-center gap-4 px-6 py-4 hover:bg-white/[0.02] transition-colors group',
                      (run.status === 'completed' || run.status === 'running') && 'cursor-pointer',
                    )}
                  >
                    {/* Status dot */}
                    <div
                      className={cn(
                        'w-2.5 h-2.5 rounded-full shrink-0',
                        STATUS_COLORS[run.status]?.dot ?? 'bg-slate-500',
                        run.status === 'running' && 'animate-pulse',
                      )}
                    />

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-semibold text-slate-200 text-sm truncate">
                          {run.model_name || 'Unknown Model'}
                        </span>
                        {run.config_json && (() => {
                          try {
                            const c = JSON.parse(run.config_json)
                            return c.training_type ? (
                              <span className="badge-neutral shrink-0">
                                {c.training_type === 'lora' ? 'QLoRA' : c.training_type}
                              </span>
                            ) : null
                          } catch { return null }
                        })()}
                      </div>
                      <div className="flex items-center gap-3 mt-1 text-xs text-slate-500">
                        {run.dataset_name && (
                          <>
                            <span className="flex items-center gap-1">
                              <Database size={11} />
                              {run.dataset_name}
                            </span>
                            <span className="w-1 h-1 rounded-full bg-slate-700" />
                          </>
                        )}
                        <span className="flex items-center gap-1">
                          <Clock size={11} />
                          {formatDate(run.started_at)}
                        </span>
                        {run.duration_seconds !== undefined && (
                          <>
                            <span className="w-1 h-1 rounded-full bg-slate-700" />
                            <span>{formatDuration(run.duration_seconds)}</span>
                          </>
                        )}
                      </div>
                    </div>

                    {/* Loss sparkline */}
                    {run.loss_sparkline && run.loss_sparkline.length > 1 && (
                      <div className="hidden md:block shrink-0">
                        <LossSparkline data={run.loss_sparkline} />
                      </div>
                    )}

                    {/* Final loss */}
                    {run.final_loss != null && (
                      <div className="hidden md:block text-right shrink-0">
                        <p className="text-xs text-slate-500">Final Loss</p>
                        <p className="text-sm font-semibold text-slate-200">{Number(run.final_loss).toFixed(4)}</p>
                      </div>
                    )}

                    {/* Status badge */}
                    <StatusBadge status={run.status} />

                    {/* Actions */}
                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      {run.status === 'running' && (
                        <button
                          onClick={e => { e.stopPropagation(); navigate('/training') }}
                          className="p-2 text-slate-500 hover:text-cap-cyan hover:bg-cap-cyan/10 rounded-lg transition-all text-xs"
                          title="View live training"
                        >
                          <Activity size={15} />
                        </button>
                      )}
                      {run.status === 'completed' && (
                        <button
                          onClick={e => { e.stopPropagation(); navigate(`/run/${run.id}`) }}
                          className="p-2 text-slate-500 hover:text-cap-cyan hover:bg-cap-cyan/10 rounded-lg transition-all"
                          title="Test in chat"
                        >
                          <MessageSquare size={15} />
                        </button>
                      )}
                      <button
                        onClick={e => { e.stopPropagation(); setDeleteConfirm(run) }}
                        className="p-2 text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-all"
                        title="Delete run"
                      >
                        <Trash2 size={15} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </motion.div>

          {/* Bottom padding */}
          <div className="h-8" />
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {deleteConfirm && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4"
          onClick={() => setDeleteConfirm(null)}
        >
          <div
            className="bg-slate-800/60 backdrop-blur-xl rounded-2xl shadow-2xl border border-white/10 p-6 max-w-md w-full"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start gap-4 mb-4">
              <div className="p-3 bg-red-500/10 rounded-xl border border-red-500/20 shrink-0">
                <Trash2 className="text-red-400" size={22} />
              </div>
              <div>
                <h3 className="text-lg font-bold text-slate-200 mb-1">Delete Training Run?</h3>
                <p className="text-slate-400 text-sm">
                  Delete{' '}
                  <span className="font-semibold text-slate-300">
                    "{deleteConfirm.model_name}"
                  </span>
                  ? This action cannot be undone.
                </p>
              </div>
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setDeleteConfirm(null)}
                disabled={deleting}
                className="flex-1 btn-secondary py-2.5 text-sm"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDeleteRun(deleteConfirm.id)}
                disabled={deleting}
                className="flex-1 btn-danger py-2.5 text-sm flex items-center justify-center gap-2"
              >
                {deleting ? (
                  <>
                    <div className="w-4 h-4 border-2 border-red-400/30 border-t-red-400 rounded-full animate-spin" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 size={14} />
                    Delete Run
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
