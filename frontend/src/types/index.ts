// ── Auth ──
export interface User {
  id?: number
  email: string
  name?: string
  role?: string
  must_change_password?: boolean
}

export interface AuthTokens {
  access_token: string
  refresh_token: string
  token_type: string
  must_change_password?: boolean
}

// ── Training Run ──
export interface TrainingRun {
  id: string
  status: 'running' | 'completed' | 'failed' | 'cancelled' | 'queued' | 'starting'
  model_name: string
  dataset_name: string
  config_json?: string
  started_at: string
  ended_at?: string
  total_steps?: number
  final_step?: number
  final_loss?: number
  output_dir?: string
  error_message?: string
  duration_seconds?: number
  loss_sparkline?: number[]
}

export interface TrainingRunListResponse {
  runs: TrainingRun[]
  total: number
}

// ── Hardware ──
export interface HardwareInfo {
  gpu_name?: string
  gpu_memory_total_gb?: number
  gpu_memory_free_gb?: number
  gpu_memory_used_gb?: number
  gpu_utilization?: number
  torch_version?: string
  cuda_version?: string
  transformers_version?: string
  unsloth_version?: string
  device?: string
}

// ── Training Config ──
export type ModelType = 'text' | 'vision' | 'audio' | 'embeddings'
export type TrainingMethod = 'qlora' | 'lora' | 'full'
export type DatasetSource = 'huggingface' | 'local'
export type GradientCheckpointing = 'unsloth' | 'true' | 'none'

export interface TrainingConfig {
  // Model
  modelType: ModelType
  modelName: string
  trainingMethod: TrainingMethod
  hfToken: string

  // Dataset
  datasetSource: DatasetSource
  datasetName: string
  datasetRows: number   // 0 = unknown; set after format check
  datasetSubset: string
  datasetSplit: string
  evalSplit: string
  formatType: 'auto' | 'alpaca' | 'chatml' | 'sharegpt' | 'custom'
  columnMapping: Record<string, string>
  systemPrompt: string
  imageColumn: string       // which column contains images (VLM datasets)
  datasetBaseDir: string    // absolute path to dataset folder (for path-only images)
  datasetSliceStart: number
  datasetSliceEnd: number

  // Core hyperparams
  numEpochs: number
  maxSteps: number // 0 = use epochs
  maxSeqLength: number
  learningRate: number
  batchSize: number
  gradAccumSteps: number
  warmupSteps: number
  weightDecay: number
  optimizer: string
  lrScheduler: string

  // LoRA
  loraR: number
  loraAlpha: number
  loraDropout: number
  targetModules: string
  useRslora: boolean
  useLoftq: boolean

  // Advanced
  packing: boolean
  trainOnCompletions: boolean
  gradientCheckpointing: GradientCheckpointing
  outputModelName: string
  saveSteps: number
  saveStrategy: 'no' | 'epoch' | 'steps' | 'best'
  earlyStoppingPatience: number
  evalSteps: number

  // Logging
  enableWandB: boolean
  wandbProject: string
  wandbToken: string
  enableTensorBoard: boolean

  // VLM specific
  finetuneVisionLayers: boolean
  finetuneLanguageLayers: boolean
  finetuneAttentionModules: boolean
  finetuneMlpModules: boolean
}

// ═══════════════════════════════════════════════════════════════
// Optimal defaults for <7B models with QLoRA/LoRA
//
// Rationale for each choice:
//   r=32, alpha=32:  Best balance for <7B. r=16 for fast experiments, r=64 for max quality.
//   cosine schedule: Better final loss than linear (gradual decay to near-zero LR).
//   epochs=1:        Safe first run. Increase to 2-3 for small datasets (<5K rows).
//   batch=2, accum=4: Effective batch=8. Sweet spot for GPU utilization vs VRAM.
//   packing=false:   Safe default. Enable for short Q&A datasets (30-50% speedup).
//   train_on_completions=true: Only learn to generate responses, not predict instructions.
//   rslora=true:     Free quality boost at r≥32. Uses √r scaling for stable gradients.
//   weight_decay=0.01: Standard regularization — prevents overfitting on small datasets.
// ═══════════════════════════════════════════════════════════════
export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  modelType: 'text',
  modelName: '',
  trainingMethod: 'qlora',
  hfToken: '',

  datasetSource: 'huggingface',
  datasetName: '',
  datasetRows: 0,
  datasetSubset: '',
  datasetSplit: 'train',
  evalSplit: '',
  formatType: 'auto',
  columnMapping: {},
  systemPrompt: '',
  imageColumn: '',
  datasetBaseDir: '',
  datasetSliceStart: 0,
  datasetSliceEnd: 0,

  numEpochs: 1,
  maxSteps: 0,
  maxSeqLength: 2048,
  learningRate: 2e-4,
  batchSize: 2,
  gradAccumSteps: 4,
  warmupSteps: 5,
  weightDecay: 0.01,
  optimizer: 'adamw_8bit',
  lrScheduler: 'cosine',       // cosine > linear for most fine-tuning tasks

  loraR: 32,                    // sweet spot for <7B models
  loraAlpha: 32,                // 1:1 with rank (standard). Try 2×r for stronger effect.
  loraDropout: 0,
  targetModules: 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
  useRslora: true,              // free quality at r≥32 — uses √r scaling
  useLoftq: false,

  packing: false,               // enable for short Q&A datasets (30-50% faster)
  trainOnCompletions: true,     // only learn responses, not instructions
  gradientCheckpointing: 'unsloth',
  outputModelName: '',
  saveSteps: 0,
  saveStrategy: 'no',
  earlyStoppingPatience: 3,
  evalSteps: 0,

  enableWandB: false,
  wandbProject: 'unslothcraft-training',
  wandbToken: '',
  enableTensorBoard: false,

  finetuneVisionLayers: true,
  finetuneLanguageLayers: true,
  finetuneAttentionModules: true,
  finetuneMlpModules: true,
}

// ── Model ──
export interface ModelInfo {
  id: string
  name: string
  path?: string
  is_local?: boolean
  is_gguf?: boolean
  is_vision?: boolean
  is_audio?: boolean
  is_embedding?: boolean
  size_gb?: number
  quant_type?: string
}

export interface LoraInfo {
  id: string
  name: string
  path: string
  base_model?: string
  training_run_id?: string
}

// ── Dataset ──
export interface DatasetCheckResult {
  format: string
  needs_mapping: boolean
  columns: string[]
  preview_rows: Record<string, unknown>[]
  suggested_mapping?: Record<string, string>
}
