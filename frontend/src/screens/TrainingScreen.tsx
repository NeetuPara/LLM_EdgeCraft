import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  StopCircle, MessageSquare, Download,
  Cpu, Activity, Clock, TrendingDown,
  CheckCircle, AlertCircle, Loader,
} from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from 'recharts'
import { motion } from 'framer-motion'
import NavBar from '@/components/NavBar'
import { useTrainingRuntimeStore } from '@/stores/training-runtime-store'
import { useTrainingConfigStore } from '@/stores/training-config-store'
import { useDemoTrainingSimulator } from '@/hooks/use-training-simulator'
import { useRealTrainingPoller } from '@/hooks/use-real-training-poller'
import { trainingApi } from '@/api/training-api'
import { isMockMode } from '@/api/mock'
import { cn } from '@/utils/cn'
import type { MetricPoint } from '@/stores/training-runtime-store'

// ── ETA formatter ──
function formatEta(seconds: number): string {
  if (seconds <= 0) return '—'
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
}

// ── Custom tooltip for charts ──
function ChartTooltip({ active, payload, label, valueLabel, decimals = 4 }: {
  active?: boolean; payload?: Array<{ value?: number }>; label?: string | number
  valueLabel?: string; decimals?: number
}) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800/95 border border-white/10 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-500 mb-0.5">Step {label}</p>
      <p className="text-white font-medium">{valueLabel || ''} {Number(payload[0]?.value ?? 0).toFixed(decimals)}</p>
    </div>
  )
}

// ── Metric chart card ──
function ChartCard({
  title, data, color, valueLabel, decimals, emptyText, yAxisLabel, xAxisLabel, yUnit,
}: {
  title: string; data: MetricPoint[]; color: string
  valueLabel?: string; decimals?: number; emptyText?: string
  yAxisLabel?: string; xAxisLabel?: string; yUnit?: string
}) {
  const hasData = data.length >= 2
  const lastVal = hasData ? Number(data[data.length - 1]?.value ?? 0) : null

  return (
    <div className="glass-card p-4 flex flex-col gap-2">
      {/* Header: title + current value */}
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-slate-300">{title}</h3>
          {yAxisLabel && (
            <p className="text-[10px] text-slate-600 mt-0.5">
              Y: {yAxisLabel} &nbsp;·&nbsp; X: {xAxisLabel ?? 'Training Step'}
            </p>
          )}
        </div>
        {lastVal !== null && (
          <span className="text-sm font-mono shrink-0" style={{ color }}>
            {lastVal.toFixed(decimals ?? 4)}{yUnit ? ` ${yUnit}` : ''}
          </span>
        )}
      </div>

      {/* Chart */}
      <div className="h-36">
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 4, right: 8, bottom: 20, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
              <XAxis
                dataKey="step"
                stroke="#475569"
                tick={{ fontSize: 9, fill: '#475569' }}
                tickLine={false}
                axisLine={false}
                label={{
                  value: xAxisLabel ?? 'Step',
                  position: 'insideBottom',
                  offset: -10,
                  fontSize: 9,
                  fill: '#475569',
                }}
              />
              <YAxis
                stroke="#475569"
                tick={{ fontSize: 9, fill: '#475569' }}
                tickLine={false}
                axisLine={false}
                width={50}
                domain={['auto', 'auto']}
                tickFormatter={(v: number) => {
                  if (decimals === 6) return v.toExponential(1)
                  return v.toFixed(2)
                }}
              />
              <Tooltip
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                content={(props: any) => (
                  <ChartTooltip {...props} valueLabel={valueLabel} decimals={decimals} />
                )}
              />
              <Line
                type="monotone"
                dataKey="value"
                stroke={color}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
                activeDot={{ r: 3, fill: color }}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full flex items-center justify-center">
            <p className="text-xs text-slate-600">{emptyText ?? 'Waiting for data...'}</p>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Log console ──
function LogConsole({ lines }: { lines: string[] }) {
  const bottomRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [lines.length])

  return (
    <div className="glass-card p-0 overflow-hidden">
      <div className="px-5 py-3 border-b border-white/[0.06] flex items-center gap-2">
        <div className="flex gap-1.5">
          <div className="w-3 h-3 rounded-full bg-red-500/60" />
          <div className="w-3 h-3 rounded-full bg-amber-500/60" />
          <div className="w-3 h-3 rounded-full bg-emerald-500/60" />
        </div>
        <span className="text-xs text-slate-500 ml-1">Training Log</span>
      </div>
      <div className="h-40 overflow-y-auto bg-slate-950/60 p-4 font-mono text-xs leading-relaxed">
        {lines.length === 0 && (
          <span className="text-slate-700">Waiting for output...</span>
        )}
        {lines.map((line, i) => (
          <div key={i} className="text-slate-400 hover:text-slate-300 transition-colors">
            <span className="text-slate-700 select-none mr-2">{String(i + 1).padStart(3, ' ')} │</span>
            {line}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

export default function TrainingScreen() {
  const navigate = useNavigate()
  const config = useTrainingConfigStore()
  const rt = useTrainingRuntimeStore()

  // Live GPU stats (real mode only — polled every 4s during training)
  const [gpuStats, setGpuStats] = useState<{
    gpu_name?: string | null
    vram_used_gb?: number | null
    vram_total_gb?: number | null
    gpu_utilization?: number | null
  } | null>(null)

  useEffect(() => {
    if (isMockMode()) return
    // Initial fetch
    Promise.all([
      trainingApi.hardware().catch(() => null),
      trainingApi.systemHardware?.().catch(() => null),
    ]).then(([hw, sys]) => {
      setGpuStats({
        gpu_name:        sys?.gpu_name        ?? hw?.gpu_name        ?? null,
        vram_used_gb:    sys?.gpu_memory_used_gb ?? null,
        vram_total_gb:   sys?.gpu_memory_total_gb ?? null,
        gpu_utilization: hw?.gpu_utilization  ?? null,
      })
    })
    // Poll utilization every 4s
    const id = setInterval(() => {
      trainingApi.hardware().then(hw => {
        setGpuStats(prev => prev ? { ...prev, gpu_utilization: hw?.gpu_utilization ?? prev.gpu_utilization } : prev)
      }).catch(() => {})
    }, 4000)
    return () => clearInterval(id)
  }, [])

  // Demo mode: simulated training data
  useDemoTrainingSimulator()
  // Real mode: polls GET /api/train/status every 1.5s
  useRealTrainingPoller()

  const isActive = rt.phase === 'running' || rt.phase === 'starting'
  const isDone = rt.phase === 'completed'
  const isFailed = rt.phase === 'failed'

  const handleStop = async () => {
    await trainingApi.stop(true)
    rt.setPhase('cancelled')
  }

  // Phase icon
  const PhaseIcon = isDone ? CheckCircle : isFailed ? AlertCircle : isActive ? Activity : Loader

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <NavBar />

      <div className="flex-1 overflow-y-auto min-h-0">
        <div className="max-w-5xl mx-auto px-6 py-6 space-y-5">

          {/* ── Header ── */}
          <motion.div
            initial={{ opacity: 1, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="glass-card p-5"
          >
            {/* Model + status row */}
            <div className="flex items-start justify-between gap-4 flex-wrap mb-4">
              <div>
                <div className="flex items-center gap-2.5 mb-1">
                  <PhaseIcon
                    size={16}
                    className={cn(
                      isDone ? 'text-emerald-400' :
                      isFailed ? 'text-red-400' :
                      isActive ? 'text-cap-cyan animate-pulse' : 'text-slate-500',
                    )}
                  />
                  <h1 className="text-lg font-bold text-slate-100 font-display truncate max-w-sm">
                    {config.modelName || 'Training Run'}
                  </h1>
                  <span className="badge-neutral text-[10px]">
                    {config.trainingMethod.toUpperCase()}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-xs text-slate-500 flex-wrap">
                  {config.datasetName && <span>{config.datasetName}</span>}
                  {rt.currentEpoch > 0 && (
                    <span className="text-slate-600">·</span>
                  )}
                  {rt.currentEpoch > 0 && (
                    <span>Epoch {rt.currentEpoch}/{rt.totalEpochs || config.numEpochs}</span>
                  )}
                  {rt.statusMessage && (
                    <>
                      <span className="text-slate-600">·</span>
                      <span className="text-slate-500 italic">{rt.statusMessage}</span>
                    </>
                  )}
                </div>
              </div>

              {/* Controls */}
              <div className="flex items-center gap-2 shrink-0">
                {isDone && (
                  <>
                    <button
                      onClick={() => navigate('/chat')}
                      className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-cap-cyan/10 border border-cap-cyan/20 text-cap-cyan text-xs font-medium hover:bg-cap-cyan/20 transition-colors"
                    >
                      <MessageSquare size={13} />
                      Test in Chat
                    </button>
                    <button
                      onClick={() => navigate('/export')}
                      className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-medium hover:bg-emerald-500/20 transition-colors"
                    >
                      <Download size={13} />
                      Export
                    </button>
                  </>
                )}
                {isActive && (
                  <button
                    onClick={handleStop}
                    className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs font-medium hover:bg-red-500/20 transition-colors"
                  >
                    <StopCircle size={13} />
                    Stop Training
                  </button>
                )}
              </div>
            </div>

            {/* Progress bar */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">
                  {rt.currentStep > 0 ? `Step ${rt.currentStep.toLocaleString()} / ${rt.totalSteps.toLocaleString()}` : 'Preparing...'}
                </span>
                <div className="flex items-center gap-4 text-slate-500">
                  {rt.loss !== null && (
                    <span className="flex items-center gap-1">
                      <TrendingDown size={11} />
                      {Number(rt.loss ?? 0).toFixed(4)}
                    </span>
                  )}
                  {rt.etaSeconds > 0 && (
                    <span className="flex items-center gap-1">
                      <Clock size={11} />
                      {formatEta(rt.etaSeconds)}
                    </span>
                  )}
                  <span className="text-cap-cyan font-semibold">{rt.progressPercent}%</span>
                </div>
              </div>
              <div className="h-2.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className={cn(
                    'h-full rounded-full transition-all duration-700',
                    isDone ? 'bg-emerald-400' : 'bg-cap-cyan',
                    isActive && rt.progressPercent > 0 && 'animate-glow-pulse',
                  )}
                  style={{ width: `${Math.max(rt.progressPercent, isActive ? 2 : 0)}%` }}
                />
              </div>
            </div>
          </motion.div>

          {/* ── GPU Stats ── */}
          <motion.div
            initial={{ opacity: 1, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.05 }}
            className="glass-card p-4"
          >
            <div className="flex items-center gap-6 flex-wrap">
              <div className="flex items-center gap-2">
                <Cpu size={14} className="text-cap-cyan" />
                <span className="text-sm font-medium text-slate-300">
                  {isMockMode() ? 'NVIDIA RTX 4090' : (gpuStats?.gpu_name ?? 'GPU')}
                </span>
              </div>
              {(() => {
                const used  = isMockMode() ? 8.2  : (gpuStats?.vram_used_gb  ?? null)
                const total = isMockMode() ? 24   : (gpuStats?.vram_total_gb ?? null)
                const pct   = (used != null && total != null && total > 0)
                  ? Math.round((used / total) * 100) : (isMockMode() ? 34 : null)
                return (
                  <div className="flex items-center gap-2 flex-1 min-w-[200px]">
                    <span className="text-xs text-slate-500 shrink-0">VRAM</span>
                    <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-cap-cyan rounded-full transition-all duration-700"
                        style={{ width: pct != null ? `${pct}%` : '0%' }} />
                    </div>
                    <span className="text-xs text-slate-400 shrink-0">
                      {used != null && total != null
                        ? `${used.toFixed(1)} / ${total.toFixed(0)} GB`
                        : total != null ? `? / ${total.toFixed(0)} GB` : '—'}
                    </span>
                  </div>
                )
              })()}
              <div className="flex items-center gap-1.5">
                <Activity size={13} className="text-slate-500" />
                <span className="text-xs text-slate-400">
                  {isMockMode() ? '78% util'
                    : gpuStats?.gpu_utilization != null
                      ? `${gpuStats.gpu_utilization}% util`
                      : '—'}
                </span>
              </div>
            </div>
          </motion.div>

          {/* ── Charts 2×2 grid ── */}
          <motion.div
            initial={{ opacity: 1, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
            className="grid grid-cols-1 sm:grid-cols-2 gap-4"
          >
            <ChartCard
              title="Training Loss"
              data={rt.lossHistory}
              color="var(--chart-loss)"
              valueLabel="loss:"
              decimals={4}
              emptyText="Training not started yet"
              yAxisLabel="Loss (lower = better)"
              xAxisLabel="Step"
            />
            <ChartCard
              title="Eval Loss"
              data={rt.evalLossHistory}
              color="var(--chart-eval-loss)"
              valueLabel="eval:"
              decimals={4}
              emptyText="No eval split configured"
              yAxisLabel="Loss on unseen data"
              xAxisLabel="Step"
            />
            <ChartCard
              title="Learning Rate"
              data={rt.lrHistory}
              color="var(--chart-lr)"
              valueLabel="lr:"
              decimals={6}
              yAxisLabel="Step size (warmup → decay)"
              xAxisLabel="Step"
            />
            <ChartCard
              title="Gradient Norm"
              data={rt.gradNormHistory}
              color="var(--chart-grad-norm)"
              valueLabel="grad norm:"
              decimals={4}
              yAxisLabel="Model confusion (high→low = good)"
              xAxisLabel="Step"
            />
          </motion.div>

          {/* ── Log Console ── */}
          <motion.div
            initial={{ opacity: 1, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.15 }}
          >
            <LogConsole lines={rt.logLines} />
          </motion.div>

          {/* ── Completed banner ── */}
          {isDone && (
            <motion.div
              initial={{ opacity: 0, scale: 0.97 }}
              animate={{ opacity: 1, scale: 1 }}
              className="glass-card p-5 border border-emerald-500/20 bg-emerald-500/5"
            >
              <div className="flex items-center gap-3">
                <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
                  <CheckCircle size={20} className="text-emerald-400" />
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-slate-200 mb-0.5">Training Complete!</h3>
                  <p className="text-sm text-slate-400">
                    {rt.outputDir
                      ? <>Model saved to <code className="text-emerald-400 text-xs break-all">{rt.outputDir}</code></>
                      : 'Model saved to outputs directory'}
                    {rt.loss != null && ` · Final loss: ${Number(rt.loss).toFixed(4)}`}
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => navigate('/chat')}
                    className="btn-primary py-2 text-sm flex items-center gap-1.5"
                  >
                    <MessageSquare size={14} />
                    Test in Chat
                  </button>
                  <button
                    onClick={() => navigate('/export')}
                    className="btn-secondary py-2 text-sm flex items-center gap-1.5"
                  >
                    <Download size={14} />
                    Export
                  </button>
                </div>
              </div>
            </motion.div>
          )}

          <div className="h-4" />
        </div>
      </div>
    </div>
  )
}
