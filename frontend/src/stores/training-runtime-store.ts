import { create } from 'zustand'

export type TrainingPhase =
  | 'idle' | 'starting' | 'running' | 'stopping'
  | 'completed' | 'failed' | 'cancelled'

export interface MetricPoint { step: number; value: number }

interface TrainingRuntimeState {
  phase: TrainingPhase
  currentStep: number
  totalSteps: number
  currentEpoch: number
  totalEpochs: number
  progressPercent: number
  etaSeconds: number
  loss: number | null
  evalLoss: number | null
  learningRate: number | null
  gradNorm: number | null
  statusMessage: string
  elapsedSeconds: number

  // Time-series for charts
  lossHistory: MetricPoint[]
  evalLossHistory: MetricPoint[]
  lrHistory: MetricPoint[]
  gradNormHistory: MetricPoint[]
  logLines: string[]
  outputDir: string | null   // set when training completes

  // Actions
  setPhase: (phase: TrainingPhase) => void
  setOutputDir: (dir: string | null) => void
  applyProgress: (data: {
    current_step?: number; total_steps?: number
    current_epoch?: number; total_epochs?: number
    progress_percent?: number; eta_seconds?: number
    loss?: number; eval_loss?: number
    learning_rate?: number; grad_norm?: number
    status_message?: string
  }) => void
  appendLog: (line: string) => void
  hydrateFromStatus: (status: {
    phase: TrainingPhase
    current_step?: number; total_steps?: number
    current_epoch?: number; total_epochs?: number
    progress_percent?: number; eta_seconds?: number
    loss?: number; eval_loss?: number
    learning_rate?: number; grad_norm?: number
    status_message?: string
    metrics?: {
      step: number[]; loss: number[]
      eval_loss: number[]; eval_steps: number[]   // eval_steps = X positions for eval_loss
      learning_rate: number[]; grad_norm: number[]
    }
  }) => void
  reset: () => void
}

const INITIAL: Omit<TrainingRuntimeState, 'setPhase' | 'applyProgress' | 'appendLog' | 'hydrateFromStatus' | 'reset'> = {
  phase: 'idle',
  currentStep: 0, totalSteps: 0,
  currentEpoch: 0, totalEpochs: 0,
  progressPercent: 0, etaSeconds: 0,
  loss: null, evalLoss: null,
  learningRate: null, gradNorm: null,
  statusMessage: '', elapsedSeconds: 0,
  lossHistory: [], evalLossHistory: [],
  lrHistory: [], gradNormHistory: [],
  logLines: [],
  outputDir: null,
}

export const useTrainingRuntimeStore = create<TrainingRuntimeState>()((set) => ({
  ...INITIAL,

  setPhase: (phase) => set({ phase }),
  setOutputDir: (outputDir) => set({ outputDir }),

  applyProgress: (data) => set((s) => {
    const step = data.current_step ?? s.currentStep
    const newState: Partial<TrainingRuntimeState> = {
      currentStep:     step,
      totalSteps:      data.total_steps ?? s.totalSteps,
      currentEpoch:    data.current_epoch ?? s.currentEpoch,
      totalEpochs:     data.total_epochs ?? s.totalEpochs,
      progressPercent: data.progress_percent ?? s.progressPercent,
      etaSeconds:      data.eta_seconds ?? s.etaSeconds,
      loss:            data.loss ?? s.loss,
      evalLoss:        data.eval_loss ?? s.evalLoss,
      learningRate:    data.learning_rate ?? s.learningRate,
      gradNorm:        data.grad_norm ?? s.gradNorm,
      statusMessage:   data.status_message ?? s.statusMessage,
    }

    if (data.loss !== undefined)
      newState.lossHistory = [...s.lossHistory, { step, value: data.loss }].slice(-200)
    if (data.eval_loss !== undefined && data.eval_loss > 0)
      newState.evalLossHistory = [...s.evalLossHistory, { step, value: data.eval_loss }].slice(-50)
    if (data.learning_rate !== undefined)
      newState.lrHistory = [...s.lrHistory, { step, value: data.learning_rate }].slice(-200)
    if (data.grad_norm !== undefined)
      newState.gradNormHistory = [...s.gradNormHistory, { step, value: data.grad_norm }].slice(-200)

    return newState
  }),

  appendLog: (line) => set((s) => ({
    logLines: [...s.logLines, line].slice(-200),
  })),

  hydrateFromStatus: (status) => set((_s) => {
    const metrics = status.metrics
    const lossHistory: MetricPoint[] = metrics
      ? metrics.step.map((s, i) => ({ step: s, value: metrics.loss[i] })).filter(p => p.value > 0)
      : []
    // eval_steps and eval_loss are parallel sparse arrays (same length, ≠ step array).
    // Using metrics.step as the index source maps eval values to the WRONG steps.
    const evalSteps: number[]   = metrics?.eval_steps ?? []
    const evalVals:  number[]   = metrics?.eval_loss   ?? []
    const evalLossHistory: MetricPoint[] = evalSteps
      .map((s: number, i: number) => ({ step: s, value: evalVals[i] }))
      .filter((p: MetricPoint) => p.value != null && isFinite(p.value as number) && (p.value as number) > 0)
    const lrHistory: MetricPoint[] = metrics
      ? metrics.step.map((s, i) => ({ step: s, value: metrics.learning_rate[i] }))
      : []
    const gradNormHistory: MetricPoint[] = metrics
      ? metrics.step.map((s, i) => ({ step: s, value: metrics.grad_norm[i] }))
      : []

    return {
      phase:           status.phase,
      currentStep:     status.current_step ?? 0,
      totalSteps:      status.total_steps ?? 0,
      currentEpoch:    status.current_epoch ?? 0,
      totalEpochs:     status.total_epochs ?? 0,
      progressPercent: status.progress_percent ?? 0,
      etaSeconds:      status.eta_seconds ?? 0,
      loss:            status.loss ?? null,
      evalLoss:        status.eval_loss ?? null,
      learningRate:    status.learning_rate ?? null,
      gradNorm:        status.grad_norm ?? null,
      statusMessage:   status.status_message ?? '',
      lossHistory, evalLossHistory, lrHistory, gradNormHistory,
    }
  }),

  reset: () => set(INITIAL),
}))
