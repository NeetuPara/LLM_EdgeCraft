// ── API ──
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8888'

// ── Pipeline Stages (LLM fine-tuning workflow) ──
export interface PipelineStage {
  id: string
  label: string
  shortLabel: string
  screens: string[]
  description: string
}

export const PIPELINE_STAGES: PipelineStage[] = [
  {
    id: 'model',
    label: 'Model Selection',
    shortLabel: 'Model',
    screens: ['/new/model'],
    description: 'Choose your base model and training method',
  },
  {
    id: 'dataset',
    label: 'Dataset',
    shortLabel: 'Dataset',
    screens: ['/new/dataset'],
    description: 'Configure your training dataset',
  },
  {
    id: 'params',
    label: 'Hyperparameters',
    shortLabel: 'Params',
    screens: ['/new/params'],
    description: 'Tune training hyperparameters',
  },
  {
    id: 'summary',
    label: 'Summary',
    shortLabel: 'Launch',
    screens: ['/new/summary'],
    description: 'Review and launch training',
  },
  {
    id: 'training',
    label: 'Training',
    shortLabel: 'Training',
    screens: ['/training'],
    description: 'Live training progress',
  },
  {
    id: 'chat',
    label: 'Chat & Test',
    shortLabel: 'Chat',
    screens: ['/chat'],
    description: 'Test your fine-tuned model',
  },
  {
    id: 'export',
    label: 'Export',
    shortLabel: 'Export',
    screens: ['/export'],
    description: 'Export in your preferred format',
  },
]

// ── Training Method Options ──
export const TRAINING_METHODS = [
  {
    id: 'qlora',
    label: 'QLoRA',
    description: '4-bit quantized base model + LoRA adapters',
    badge: '4-bit',
    recommended: true,
    vramNote: 'Lowest VRAM (~4–8 GB)',
  },
  {
    id: 'lora',
    label: 'LoRA',
    description: '16-bit base model + LoRA adapters',
    badge: '16-bit',
    recommended: false,
    vramNote: 'Medium VRAM (~10–20 GB)',
  },
  {
    id: 'full',
    label: 'Full Fine-tune',
    description: 'All parameters trained (no LoRA)',
    badge: 'Full',
    recommended: false,
    vramNote: 'High VRAM (24 GB+)',
  },
] as const

// ── Export Methods ──
export const EXPORT_METHODS = [
  {
    id: 'lora',
    label: 'LoRA Adapter',
    description: 'Adapter weights only — smallest size, requires base model to run',
    icon: 'link',
    sizeNote: '~50–200 MB',
  },
  {
    id: 'merged_16bit',
    label: 'Merged 16-bit',
    description: 'Base model + LoRA merged into float16 — ready to run standalone',
    icon: 'merge',
    sizeNote: 'Same as base model',
  },
  {
    id: 'gguf',
    label: 'GGUF',
    description: 'Quantized for Ollama, LM Studio, llama.cpp',
    icon: 'package',
    sizeNote: 'Variable by quant level',
  },
] as const

// ── GGUF Quantization Options ──
export const GGUF_QUANT_OPTIONS = [
  { id: 'q4_k_m', label: 'Q4_K_M', description: 'Recommended — best balance', recommended: true },
  { id: 'q5_k_m', label: 'Q5_K_M', description: 'Higher quality, larger size', recommended: false },
  { id: 'q8_0',   label: 'Q8_0',   description: 'Near-lossless, largest',       recommended: false },
  { id: 'f16',    label: 'F16',    description: 'No quantization — float16',     recommended: false },
  { id: 'q3_k_m', label: 'Q3_K_M', description: 'Smaller, lower quality',       recommended: false },
  { id: 'q2_k',   label: 'Q2_K',   description: 'Smallest, lowest quality',     recommended: false },
] as const

// ── Model Types ──
export const MODEL_TYPES = [
  { id: 'text',       label: 'Text',       description: 'Language models for text generation & chat',       icon: 'message-square' },
  { id: 'vision',     label: 'Vision',     description: 'Vision-language models for image + text',          icon: 'eye' },
  { id: 'audio',      label: 'Audio',      description: 'Speech & audio models',                            icon: 'mic' },
  { id: 'embeddings', label: 'Embeddings', description: 'Sentence transformers & embedding models',         icon: 'box' },
] as const

// ── LR Schedulers ──
export const LR_SCHEDULERS = [
  { id: 'linear',          label: 'Linear' },
  { id: 'cosine',          label: 'Cosine' },
  { id: 'cosine_with_restarts', label: 'Cosine w/ Restarts' },
  { id: 'polynomial',      label: 'Polynomial' },
  { id: 'constant',        label: 'Constant' },
  { id: 'constant_with_warmup', label: 'Constant w/ Warmup' },
] as const

// ── Optimizers ──
export const OPTIMIZERS = [
  { id: 'adamw_8bit',  label: 'AdamW 8-bit (recommended)' },
  { id: 'adamw_torch', label: 'AdamW (fp32)' },
  { id: 'sgd',         label: 'SGD' },
  { id: 'adagrad',     label: 'Adagrad' },
  { id: 'adamw_torch_fused', label: 'AdamW Fused' },
] as const

// ── Training Status Colors ──
export const STATUS_COLORS: Record<string, { bg: string; text: string; border: string; dot: string }> = {
  running:   { bg: 'bg-cap-cyan/10',     text: 'text-cap-cyan',     border: 'border-cap-cyan/20',     dot: 'bg-cap-cyan' },
  completed: { bg: 'bg-emerald-500/10',  text: 'text-emerald-400',  border: 'border-emerald-500/20',  dot: 'bg-emerald-400' },
  failed:    { bg: 'bg-red-500/10',      text: 'text-red-400',      border: 'border-red-500/20',      dot: 'bg-red-400' },
  cancelled: { bg: 'bg-slate-800',       text: 'text-slate-500',    border: 'border-slate-700',       dot: 'bg-slate-500' },
  queued:    { bg: 'bg-amber-500/10',    text: 'text-amber-400',    border: 'border-amber-500/20',    dot: 'bg-amber-400' },
  starting:  { bg: 'bg-indigo-500/10',   text: 'text-indigo-400',   border: 'border-indigo-500/20',   dot: 'bg-indigo-400' },
}
