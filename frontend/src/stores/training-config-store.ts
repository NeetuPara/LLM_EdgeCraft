import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { TrainingConfig } from '@/types'
import { DEFAULT_TRAINING_CONFIG } from '@/types'

interface TrainingConfigState extends TrainingConfig {
  // Wizard step tracking
  highestStep: number
  setHighestStep: (step: number) => void

  // Patch config
  patch: (updates: Partial<TrainingConfig>) => void
  reset: () => void
}

export const useTrainingConfigStore = create<TrainingConfigState>()(
  persist(
    (set) => ({
      ...DEFAULT_TRAINING_CONFIG,
      highestStep: 0,

      setHighestStep: (step) =>
        set((s) => ({ highestStep: Math.max(s.highestStep, step) })),

      patch: (updates) => set((s) => ({ ...s, ...updates })),

      reset: () =>
        set({ ...DEFAULT_TRAINING_CONFIG, highestStep: 0 }),
    }),
    {
      name: 'unslothcraft-training-config-v1',
      version: 1,
    },
  ),
)
