import { apiFetch } from './client'
import { isMockMode, mockTraining, mockSystem } from './mock'
import type { TrainingRun, TrainingRunListResponse, HardwareInfo } from '@/types'

export interface TrainingStartRequest {
  model_name: string
  // Field names must match backend's Pydantic model exactly
  training_type?: string            // "LoRA/QLoRA" for LoRA, "full" for full finetune
  hf_token?: string
  load_in_4bit?: boolean
  max_seq_length?: number
  hf_dataset?: string               // HuggingFace dataset ID (not "dataset")
  local_datasets?: string[]          // Local file paths (not "local_dataset")
  format_type?: string
  subset?: string
  train_split?: string
  eval_split?: string
  eval_steps?: number
  dataset_slice_start?: number
  dataset_slice_end?: number
  custom_format_mapping?: Record<string, string>
  system_prompt?: string
  image_column?: string
  dataset_base_dir?: string
  is_dataset_image?: boolean
  is_dataset_audio?: boolean
  is_embedding?: boolean
  num_epochs?: number
  learning_rate?: string | number
  batch_size?: number
  gradient_accumulation_steps?: number
  warmup_steps?: number
  warmup_ratio?: number
  max_steps?: number
  save_steps?: number
  weight_decay?: number
  random_seed?: number
  packing?: boolean
  optim?: string
  lr_scheduler_type?: string
  use_lora?: boolean
  lora_r?: number
  lora_alpha?: number
  lora_dropout?: number
  target_modules?: string
  gradient_checkpointing?: string
  use_rslora?: boolean
  use_loftq?: boolean
  train_on_completions?: boolean
  finetune_vision_layers?: boolean
  finetune_language_layers?: boolean
  finetune_attention_modules?: boolean
  finetune_mlp_modules?: boolean
  enable_wandb?: boolean
  wandb_token?: string
  wandb_project?: string
  enable_tensorboard?: boolean
  tensorboard_dir?: string
  trust_remote_code?: boolean
  gpu_ids?: number[]
}

export interface TrainingStatus {
  phase: 'idle' | 'starting' | 'running' | 'stopping' | 'completed' | 'failed' | 'cancelled'
  is_training: boolean
  current_step?: number
  total_steps?: number
  current_epoch?: number
  total_epochs?: number
  progress_percent?: number
  eta_seconds?: number
  loss?: number
  eval_loss?: number
  learning_rate?: number
  grad_norm?: number
  status_message?: string
  error?: string
  output_dir?: string | null
  metric_history?: {
    steps: number[]
    loss: number[]
    eval_loss: number[]
    lr: number[]
    grad_norm: number[]
    grad_norm_steps: number[]
    eval_steps: number[]
  }
  metrics?: {
    step: number[]
    loss: number[]
    eval_loss: number[]
    learning_rate: number[]
    grad_norm: number[]
    epoch: number[]
  }
}

export const trainingApi = {
  start: (config: TrainingStartRequest) =>
    isMockMode()
      ? mockTraining.start()
      : apiFetch<{ job_id: string; status: string }>('/api/train/start', {
          method: 'POST',
          body: JSON.stringify(config),
        }),

  stop: (save = true) =>
    isMockMode()
      ? mockTraining.stop()
      : apiFetch<void>('/api/train/stop', {
          method: 'POST',
          body: JSON.stringify({ save }),
        }),

  reset: () =>
    isMockMode()
      ? mockTraining.reset()
      : apiFetch<void>('/api/train/reset', { method: 'POST' }),

  status: () =>
    isMockMode()
      ? mockTraining.status()
      : apiFetch<TrainingStatus>('/api/train/status'),

  metrics: () =>
    isMockMode()
      ? mockTraining.metrics()
      : apiFetch<TrainingStatus['metrics']>('/api/train/metrics'),

  hardware: () =>
    isMockMode()
      ? mockTraining.hardware()
      : apiFetch<{ gpu_utilization?: number; vram_used_mb?: number; vram_total_mb?: number; gpu_name?: string }>(
          '/api/train/hardware',
        ),

  systemHardware: () =>
    isMockMode()
      ? mockSystem.hardware()
      : apiFetch<HardwareInfo>('/api/system/hardware'),

  listRuns: (limit = 50, offset = 0) =>
    isMockMode()
      ? mockTraining.listRuns()
      : apiFetch<TrainingRunListResponse>(`/api/train/runs?limit=${limit}&offset=${offset}`),

  getRun: (id: string) =>
    isMockMode()
      ? mockTraining.getRun(id)
      : apiFetch<TrainingRun>(`/api/train/runs/${id}`),

  deleteRun: (id: string) =>
    isMockMode()
      ? mockTraining.deleteRun(id)
      : apiFetch<void>(`/api/train/runs/${id}`, { method: 'DELETE' }),
}

export const systemApi = {
  health: () =>
    isMockMode()
      ? mockSystem.health()
      : apiFetch<{ status: string; platform: string; chat_only: boolean }>('/api/health', {}, true),

  hardware: () =>
    isMockMode()
      ? mockSystem.hardware()
      : apiFetch<HardwareInfo>('/api/system/hardware'),
}
