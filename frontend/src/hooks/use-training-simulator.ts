// Demo mode training simulator — produces realistic live training data
// without a real backend. Only active when VITE_DEMO_MODE=true.

import { useEffect, useRef } from 'react'
import { isMockMode } from '@/api/mock'
import { useTrainingRuntimeStore } from '@/stores/training-runtime-store'
import { MOCK_TRAINING_STATUS } from '@/api/mock/data'

const TOTAL_STEPS = 400   // ~5 min demo at 1.5s/tick, +2 steps/tick
const TOTAL_EPOCHS = 1

const DEMO_LOGS = [
  'Loading model unsloth/Gemma-3-4B-Instruct...',
  'Model loaded in 8.3s — 4-bit NF4 quantization active',
  'Loading dataset Open-Orca/OpenOrca (train split)...',
  'Dataset loaded: 2,000 samples selected',
  'Preparing model for LoRA training (r=64, alpha=16)...',
  'LoRA adapters attached to 7 target modules',
  `Starting training: ${TOTAL_STEPS} steps · ${TOTAL_EPOCHS} epoch · batch=2 · lr=2e-4`,
]

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

function addNoise(value: number, magnitude: number) {
  return value + (Math.random() - 0.5) * magnitude
}

export function useDemoTrainingSimulator() {
  const { hydrateFromStatus, applyProgress, appendLog, setPhase } = useTrainingRuntimeStore()
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const stepRef = useRef(0)          // start from 0
  const logIndexRef = useRef(0)
  const elapsedRef = useRef(0)
  const completedRef = useRef(false)

  useEffect(() => {
    if (!isMockMode()) return

    // Hydrate with initial state from mock
    hydrateFromStatus(MOCK_TRAINING_STATUS as Parameters<typeof hydrateFromStatus>[0])

    // Emit setup logs with delays
    DEMO_LOGS.forEach((log, i) => {
      setTimeout(() => appendLog(log), i * 180)
    })
    logIndexRef.current = DEMO_LOGS.length

    // Tick every 1.5s — simulate training progress
    intervalRef.current = setInterval(() => {
      if (completedRef.current) return

      // +2 steps per tick, 1.5s interval = ~5 minutes for TOTAL_STEPS
      const step = Math.min(stepRef.current + 2, TOTAL_STEPS)
      stepRef.current = step
      elapsedRef.current += 1.5

      const t = step / TOTAL_STEPS   // 0 → 1 progress
      const epoch = 1                 // demo runs 1 epoch

      // Realistic loss curve: high → smooth decay with noise
      const baseLoss = lerp(2.2, 0.62, Math.pow(t, 0.6))
      const loss = Math.max(0.5, addNoise(baseLoss, 0.04))

      // Eval loss slightly higher than train loss, only emitted every ~200 steps
      const emitEval = step % 80 < 3  // eval every 80 steps (5 eval points in 400 steps)
      const evalLoss = emitEval ? Math.max(0.55, addNoise(baseLoss + 0.08, 0.03)) : undefined

      // LR with linear warmup then linear decay
      const maxLr = 2e-4
      const warmupSteps = 5
      const lr = step < warmupSteps
        ? (step / warmupSteps) * maxLr
        : maxLr * (1 - (step - warmupSteps) / (2000 - warmupSteps))

      // Grad norm: starts high, settles
      const gradNorm = Math.max(0.3, addNoise(lerp(1.4, 0.7, t), 0.08))

      const etaSeconds = Math.max(0, Math.round(((TOTAL_STEPS - step) / 2) * 1.5))
      const progressPercent = Math.round((step / TOTAL_STEPS) * 100)

      applyProgress({
        current_step: step,
        total_steps: TOTAL_STEPS,
        current_epoch: epoch,
        total_epochs: 1,
        progress_percent: progressPercent,
        eta_seconds: etaSeconds,
        loss,
        eval_loss: evalLoss,
        learning_rate: lr,
        grad_norm: gradNorm,
        status_message: `Training epoch ${epoch}/${TOTAL_EPOCHS}...`,
      })

      // Emit log line every ~100 steps
      if (step % 40 < 3) {   // log every 40 steps
        const tokPerSec = Math.round(addNoise(1800, 100))
        appendLog(
          `Step ${step}/${TOTAL_STEPS} | loss: ${loss.toFixed(4)} | lr: ${lr.toExponential(2)} | ` +
          `grad_norm: ${gradNorm.toFixed(4)} | ${tokPerSec} tok/s`
        )
      }

      if (step >= TOTAL_STEPS) {
        completedRef.current = true
        setPhase('completed')
        appendLog('Training complete! Best checkpoint saved to ./outputs/')
        appendLog(`Final loss: ${loss.toFixed(4)} | Total time: ${Math.round(elapsedRef.current / 60)}m`)
        if (intervalRef.current) clearInterval(intervalRef.current)
      }
    }, 1500)

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
}
