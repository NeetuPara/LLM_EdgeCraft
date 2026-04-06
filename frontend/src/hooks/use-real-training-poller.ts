/**
 * Polls GET /api/train/status every 1.5s during real training (non-demo mode).
 * Updates useTrainingRuntimeStore with live metrics from the backend.
 */
import { useEffect, useRef } from 'react'
import { isMockMode } from '@/api/mock'
import { trainingApi, type TrainingStatus } from '@/api/training-api'
import { useTrainingRuntimeStore } from '@/stores/training-runtime-store'
import type { TrainingPhase } from '@/stores/training-runtime-store'

const POLL_INTERVAL_MS = 1500
const TERMINAL_PHASES: TrainingPhase[] = ['completed', 'failed', 'cancelled']

export function useRealTrainingPoller() {
  const {
    setPhase, applyProgress, appendLog, hydrateFromStatus, setOutputDir,
  } = useTrainingRuntimeStore()

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const lastStepRef = useRef(-1)
  const lastStatusMsgRef = useRef('')
  const mountedRef = useRef(true)
  const hydratedRef = useRef(false)

  useEffect(() => {
    if (isMockMode()) return
    mountedRef.current = true

    const poll = async () => {
      if (!mountedRef.current) return

      try {
        const status = await trainingApi.status() as TrainingStatus

        if (!mountedRef.current) return

        // ── Map phase ──
        const backendPhase = status.phase as string
        let mappedPhase: TrainingPhase = 'idle'
        if (['training', 'running'].includes(backendPhase)) mappedPhase = 'running'
        else if (['loading_model', 'configuring', 'starting'].includes(backendPhase)) mappedPhase = 'starting'
        else if (backendPhase === 'completed') mappedPhase = 'completed'
        else if (backendPhase === 'error') mappedPhase = 'failed'
        else if (['stopped', 'cancelled'].includes(backendPhase)) mappedPhase = 'cancelled'

        setPhase(mappedPhase)

        const step = status.current_step ?? 0
        const isActive = status.is_training || mappedPhase === 'starting'

        // ── Update metric store ──
        if (step > 0 || isActive) {
          applyProgress({
            current_step: step,
            total_steps: status.total_steps,
            current_epoch: status.current_epoch,
            progress_percent: status.progress_percent,
            eta_seconds: status.eta_seconds,
            loss: status.loss,
            eval_loss: status.eval_loss,
            learning_rate: status.learning_rate,
            grad_norm: status.grad_norm,
            status_message: status.status_message || '',
          })
        }

        // ── Hydrate full metric history once on first load ──
        if (!hydratedRef.current && status.metric_history) {
          const mh = status.metric_history
          if (mh.steps && mh.steps.length > 0) {
            hydrateFromStatus({
              phase: mappedPhase,
              current_step: step,
              total_steps: status.total_steps,
              current_epoch: status.current_epoch,
              progress_percent: status.progress_percent,
              eta_seconds: status.eta_seconds,
              loss: status.loss,
              eval_loss: status.eval_loss,
              learning_rate: status.learning_rate,
              grad_norm: status.grad_norm,
              status_message: status.status_message,
              metrics: {
                step: mh.steps ?? [],
                loss: mh.loss ?? [],
                eval_loss: mh.eval_loss ?? [],
                learning_rate: mh.lr ?? [],
                grad_norm: mh.grad_norm ?? [],
              },
            })
            hydratedRef.current = true
          }
        }

        // ── Append to log console — only when something NEW happens ──

        // 1. Status message changed (model loading, dataset loading, etc.)
        const msg = status.status_message || ''
        if (msg && msg !== lastStatusMsgRef.current && !msg.startsWith('Training step')) {
          appendLog(`▶ ${msg}`)
          lastStatusMsgRef.current = msg
        }

        // 2. New training step — append ONE line per step (not per poll)
        if (step > lastStepRef.current && step > 0 && status.loss != null) {
          const total = status.total_steps || '?'
          const loss = Number(status.loss).toFixed(4)
          const lr = status.learning_rate != null
            ? Number(status.learning_rate).toExponential(2)
            : '?'
          const epoch = status.current_epoch != null
            ? `epoch ${Number(status.current_epoch).toFixed(2)}`
            : ''
          const gradNorm = status.grad_norm != null
            ? ` | grad_norm: ${Number(status.grad_norm).toFixed(4)}`
            : ''
          appendLog(
            `Step ${step}/${total}${epoch ? ' | ' + epoch : ''} | loss: ${loss} | lr: ${lr}${gradNorm}`
          )
          lastStepRef.current = step
        }

        // 3. Error
        if (status.error && status.error !== lastStatusMsgRef.current) {
          appendLog(`❌ ERROR: ${status.error}`)
          lastStatusMsgRef.current = status.error
        }

        // 4. Completed
        if (mappedPhase === 'completed' && lastStatusMsgRef.current !== 'completed') {
          const finalLoss = status.loss != null ? ` — Final loss: ${Number(status.loss).toFixed(4)}` : ''
          appendLog(`✅ Training complete!${finalLoss}`)
          if (status.output_dir) setOutputDir(status.output_dir)
          lastStatusMsgRef.current = 'completed'
        }

        // ── Stop polling when done ──
        if (TERMINAL_PHASES.includes(mappedPhase) && step > 0) {
          if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
          }
        }

      } catch {
        // Backend unreachable — keep trying silently
      }
    }

    poll() // immediate first poll
    intervalRef.current = setInterval(poll, POLL_INTERVAL_MS)

    return () => {
      mountedRef.current = false
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
}
