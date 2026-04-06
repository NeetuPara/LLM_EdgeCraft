// ── All mock data for DEMO_MODE ──
// Realistic fake responses that mirror the real Unsloth Studio API shapes.

import type { TrainingRun, HardwareInfo } from '@/types'

export const MOCK_USER = {
  id: 1,
  email: 'demo@example.com',
  name: 'Demo User',
  role: 'user',
}

export const MOCK_AUTH_TOKENS = {
  access_token: 'demo_access_token',
  token_type: 'bearer',
  msg: 'Logged in',
  user: { id: 1, email: 'demo@example.com', name: 'Demo User', role: 'user' },
}

export const MOCK_HARDWARE: HardwareInfo = {
  gpu_name: 'NVIDIA RTX 4090',
  gpu_memory_total_gb: 24,
  gpu_memory_free_gb: 18.4,
  gpu_memory_used_gb: 5.6,
  gpu_utilization: 12,
  torch_version: '2.9.1+cu128',
  cuda_version: '12.8',
  transformers_version: '4.51.3',
  unsloth_version: '2026.3.18',
  device: 'cuda',
}

export const MOCK_RUNS: TrainingRun[] = [
  {
    id: 'run_001',
    status: 'completed',
    model_name: 'unsloth/Llama-3.2-3B-Instruct',
    dataset_name: 'yahma/alpaca-cleaned',
    config_json: JSON.stringify({ training_type: 'lora', lora_r: 64 }),
    started_at: new Date(Date.now() - 3600 * 24 * 2 * 1000).toISOString(),
    ended_at: new Date(Date.now() - 3600 * 24 * 2 * 1000 + 5400 * 1000).toISOString(),
    total_steps: 1200,
    final_step: 1200,
    final_loss: 0.8821,
    duration_seconds: 5400,
    output_dir: './outputs/run_001',
    loss_sparkline: [2.1, 1.8, 1.5, 1.3, 1.2, 1.1, 1.05, 0.99, 0.95, 0.92, 0.90, 0.88],
  },
  {
    id: 'run_002',
    status: 'completed',
    model_name: 'unsloth/Qwen2.5-7B-Instruct',
    dataset_name: 'teknium/OpenHermes-2.5',
    config_json: JSON.stringify({ training_type: 'lora', lora_r: 128 }),
    started_at: new Date(Date.now() - 3600 * 24 * 1 * 1000).toISOString(),
    ended_at: new Date(Date.now() - 3600 * 20 * 1000).toISOString(),
    total_steps: 3000,
    final_step: 3000,
    final_loss: 0.7134,
    duration_seconds: 14400,
    output_dir: './outputs/run_002',
    loss_sparkline: [2.4, 2.0, 1.7, 1.5, 1.3, 1.15, 1.05, 0.95, 0.88, 0.81, 0.76, 0.71],
  },
  {
    id: 'run_003',
    status: 'running',
    model_name: 'unsloth/Gemma-3-4B-Instruct',
    dataset_name: 'Open-Orca/OpenOrca',
    config_json: JSON.stringify({ training_type: 'lora', lora_r: 64 }),
    started_at: new Date(Date.now() - 3600 * 1.5 * 1000).toISOString(),
    total_steps: 2000,
    final_step: 840,
    duration_seconds: 5400,
    loss_sparkline: [2.2, 1.9, 1.65, 1.45, 1.3, 1.2, 1.12],
  },
  {
    id: 'run_004',
    status: 'failed',
    model_name: 'unsloth/Llama-3.1-8B',
    dataset_name: 'custom_dataset.jsonl',
    config_json: JSON.stringify({ training_type: 'full' }),
    started_at: new Date(Date.now() - 3600 * 48 * 1000).toISOString(),
    ended_at: new Date(Date.now() - 3600 * 47 * 1000).toISOString(),
    total_steps: 0,
    final_step: 0,
    duration_seconds: 180,
    error_message: 'CUDA out of memory. Required 28GB, available 24GB.',
    loss_sparkline: [],
  },
]

export const MOCK_TRAINING_STATUS = {
  phase: 'running' as const,
  is_training: true,
  current_step: 0,
  total_steps: 400,      // 1 epoch, ~5 min demo
  current_epoch: 1,
  total_epochs: 1,
  progress_percent: 0,
  eta_seconds: 300,
  loss: 2.2,
  eval_loss: undefined,
  learning_rate: 0.0,
  grad_norm: 0.0,
  status_message: 'Training epoch 2/3...',
  metrics: {
    step:          [100, 200, 300, 400, 500, 600, 700, 800, 840],
    loss:          [2.21, 1.89, 1.65, 1.45, 1.30, 1.20, 1.13, 1.18, 1.12],
    eval_loss:     [0,    0,    1.71, 0,    1.52, 0,    1.28, 0,    1.21],
    learning_rate: [2e-4, 1.9e-4, 1.8e-4, 1.7e-4, 1.6e-4, 1.5e-4, 1.48e-4, 1.47e-4, 1.48e-4],
    grad_norm:     [1.21, 1.05, 0.98, 0.91, 0.87, 0.90, 0.88, 0.85, 0.88],
    epoch:         [1, 1, 1, 1, 2, 2, 2, 2, 2],
  },
}

// Popular datasets for search
export const MOCK_DATASETS_SEARCH = [
  { id: 'yahma/alpaca-cleaned',           name: 'yahma/alpaca-cleaned',           rows: 52002,  format: 'alpaca',   desc: 'Cleaned Alpaca instruction-following dataset' },
  { id: 'teknium/OpenHermes-2.5',         name: 'teknium/OpenHermes-2.5',         rows: 1001551, format: 'sharegpt', desc: 'High-quality GPT-4 instruction dataset' },
  { id: 'Open-Orca/OpenOrca',             name: 'Open-Orca/OpenOrca',             rows: 4233923, format: 'sharegpt', desc: 'OpenOrca GPT-4 augmented FLAN dataset' },
  { id: 'tatsu-lab/alpaca',               name: 'tatsu-lab/alpaca',               rows: 52000,  format: 'alpaca',   desc: 'Stanford Alpaca instruction-following data' },
  { id: 'WizardLM/WizardLM_evol_instruct_70k', name: 'WizardLM/WizardLM_evol_instruct_70k', rows: 70000, format: 'sharegpt', desc: 'Evolved instructions from WizardLM' },
  { id: 'databricks/databricks-dolly-15k', name: 'databricks/databricks-dolly-15k', rows: 15011, format: 'alpaca',  desc: 'High-quality human-generated instruction data' },
  { id: 'HuggingFaceH4/ultrachat_200k',   name: 'HuggingFaceH4/ultrachat_200k',   rows: 200000, format: 'sharegpt', desc: 'Filtered UltraChat multi-turn conversations' },
  { id: 'garage-bAInd/Open-Platypus',     name: 'garage-bAInd/Open-Platypus',     rows: 24926,  format: 'alpaca',   desc: 'STEM & logic instruction dataset' },
  { id: 'mlabonne/guanaco-llama2-1k',     name: 'mlabonne/guanaco-llama2-1k',     rows: 1000,   format: 'sharegpt', desc: 'Small Guanaco dataset for quick testing' },
  { id: 'iamtarun/python_code_instructions_18k_alpaca', name: 'iamtarun/python_code_instructions_18k_alpaca', rows: 18000, format: 'alpaca', desc: 'Python code instruction dataset' },
]

// ── Text Models ≤4B (curated, 2023-2026) ──
// Only best-in-class from Llama / Gemma / Qwen / Mistral / Phi families
export const MOCK_TEXT_MODELS = [
  // Llama (Meta) — gated: requires accepting Meta license on HuggingFace
  { id: 't1',  name: 'unsloth/Llama-3.2-1B-Instruct',            params: '1B',   size_gb: 0.7,  org: 'Meta',        year: 2024, is_gated: true,  type: 'text', desc: 'Fast, lightweight. Good for quick experiments.' },
  { id: 't2',  name: 'unsloth/Llama-3.2-3B-Instruct',            params: '3B',   size_gb: 2.0,  org: 'Meta',        year: 2024, is_gated: true,  type: 'text', desc: 'Best balance of speed and quality at ≤4B.' },
  // Gemma (Google) — gated: requires accepting Google license
  { id: 't3',  name: 'unsloth/gemma-3-1b-it',                    params: '1B',   size_gb: 0.7,  org: 'Google',      year: 2025, is_gated: true,  type: 'text', desc: 'Latest Gemma 3, very capable at 1B.' },
  { id: 't4',  name: 'unsloth/gemma-3-4b-it',                    params: '4B',   size_gb: 2.5,  org: 'Google',      year: 2025, is_gated: true,  type: 'text', desc: 'Gemma 3 multimodal — text + vision fine-tuning.' },
  // Qwen (Alibaba) — open, no HF token needed
  { id: 't5',  name: 'unsloth/Qwen3-0.6B',                       params: '0.6B', size_gb: 0.4,  org: 'Qwen',        year: 2025, is_gated: false, type: 'text', desc: 'Smallest Qwen3. Thinking model, very fast.' },
  { id: 't6',  name: 'unsloth/Qwen3-1.7B',                       params: '1.7B', size_gb: 1.1,  org: 'Qwen',        year: 2025, is_gated: false, type: 'text', desc: 'Qwen3 with chain-of-thought reasoning.' },
  { id: 't7',  name: 'unsloth/Qwen2.5-0.5B-Instruct',            params: '0.5B', size_gb: 0.3,  org: 'Qwen',        year: 2024, is_gated: false, type: 'text', desc: 'Tiny but surprisingly capable.' },
  { id: 't8',  name: 'unsloth/Qwen2.5-1.5B-Instruct',            params: '1.5B', size_gb: 1.0,  org: 'Qwen',        year: 2024, is_gated: false, type: 'text', desc: 'Strong instruction following at 1.5B.' },
  { id: 't9',  name: 'unsloth/Qwen2.5-3B-Instruct',              params: '3B',   size_gb: 2.0,  org: 'Qwen',        year: 2024, is_gated: false, type: 'text', desc: 'Best open small model for instruction tuning.' },
  // Phi (Microsoft) — open
  { id: 't10', name: 'unsloth/Phi-4-mini-Instruct',               params: '3.8B', size_gb: 2.4,  org: 'Microsoft',   year: 2025, is_gated: false, type: 'text', desc: 'Phi-4 mini: strong reasoning, code, math.' },
  // DeepSeek — open
  { id: 't11', name: 'unsloth/DeepSeek-R1-Distill-Qwen-1.5B',    params: '1.5B', size_gb: 1.0,  org: 'DeepSeek',    year: 2025, is_gated: false, type: 'text', desc: 'R1 reasoning distilled into 1.5B.' },
  // SmolLM (HuggingFace) — open
  { id: 't12', name: 'HuggingFaceTB/SmolLM2-1.7B-Instruct',      params: '1.7B', size_gb: 1.1,  org: 'HuggingFace', year: 2024, is_gated: false, type: 'text', desc: 'Compact, efficient, great for edge devices.' },
]

// ── Vision Models ≤4B (curated, 2023-2026) ──
export const MOCK_VISION_MODELS = [
  // Gemma 3 (Google) — gated, multimodal: also in text tab
  { id: 'v1', name: 'unsloth/gemma-3-4b-it',                     params: '4B',   size_gb: 2.5,  org: 'Google',      year: 2025, is_gated: true,  type: 'vision', desc: 'Gemma 3 multimodal — text + vision fine-tuning.' },
  // Qwen Vision (Alibaba) — open
  { id: 'v2', name: 'unsloth/Qwen2.5-VL-3B-Instruct',            params: '3B',   size_gb: 2.0,  org: 'Qwen',        year: 2024, is_gated: false, type: 'vision', desc: 'Strong visual understanding + instruction following.' },
  { id: 'v3', name: 'unsloth/Qwen2-VL-2B-Instruct',              params: '2B',   size_gb: 1.4,  org: 'Qwen',        year: 2024, is_gated: false, type: 'vision', desc: 'Compact vision-language model.' },
  // PaliGemma (Google) — gated
  { id: 'v4', name: 'google/paligemma2-3b-pt-224',               params: '3B',   size_gb: 2.0,  org: 'Google',      year: 2024, is_gated: true,  type: 'vision', desc: 'PaliGemma2: Google vision model, image captioning & VQA.' },
  // SmolVLM (HuggingFace) — open
  { id: 'v5', name: 'HuggingFaceTB/SmolVLM-Instruct',            params: '2B',   size_gb: 1.3,  org: 'HuggingFace', year: 2024, is_gated: false, type: 'vision', desc: 'Tiny but capable VLM, great for fine-tuning.' },
  { id: 'v8', name: 'HuggingFaceTB/SmolVLM-500M-Instruct',       params: '500M', size_gb: 0.4,  org: 'HuggingFace', year: 2024, is_gated: false, type: 'vision', desc: 'Ultra-compact 500M VLM. Fastest to fine-tune.' },
  { id: 'v9', name: 'HuggingFaceTB/SmolVLM-500M-Base',          params: '500M', size_gb: 0.4,  org: 'HuggingFace', year: 2024, is_gated: false, type: 'vision', desc: 'Base (pre-trained only, no instruction tuning). Fine-tune from scratch.' },
  // Moondream — open
  { id: 'v6', name: 'vikhyatk/moondream2',                       params: '1.8B', size_gb: 1.1,  org: 'Moondream',   year: 2024, is_gated: false, type: 'vision', desc: 'Ultra-small vision model optimized for edge.' },
  // InternVL (OpenGVLab) — open
  { id: 'v7', name: 'OpenGVLab/InternVL2-2B',                    params: '2B',   size_gb: 1.4,  org: 'OpenGVLab',   year: 2024, is_gated: false, type: 'vision', desc: 'Strong open-source VLM from Shanghai AI Lab.' },
]

// Combined (used by ChatScreen model selector)
export const MOCK_MODELS = [...MOCK_TEXT_MODELS, ...MOCK_VISION_MODELS]

export const MOCK_LORAS = [
  {
    id: 'l1', name: 'run_001 — Llama-3.2-3B-alpaca',
    path: './outputs/run_001', base_model: 'unsloth/Llama-3.2-3B-Instruct',
    training_run_id: 'run_001',
  },
  {
    id: 'l2', name: 'run_002 — Qwen2.5-7B-hermes',
    path: './outputs/run_002', base_model: 'unsloth/Qwen2.5-7B-Instruct',
    training_run_id: 'run_002',
  },
]

export const MOCK_CHECKPOINTS = [
  { id: 'cp1', path: './outputs/run_001/checkpoint-400', step: 400,  loss: 1.12, run_id: 'run_001' },
  { id: 'cp2', path: './outputs/run_001/checkpoint-800', step: 800,  loss: 0.96, run_id: 'run_001' },
  { id: 'cp3', path: './outputs/run_001',                step: 1200, loss: 0.88, run_id: 'run_001', is_final: true },
]

export const MOCK_DATASET_CHECK = {
  format: 'alpaca',
  needs_mapping: false,
  columns: ['instruction', 'input', 'output'],
  preview_rows: [
    { instruction: 'Write a short poem about the ocean.', input: '', output: 'Waves crash and spray,\nSalt on the morning breeze...' },
    { instruction: 'Explain neural networks simply.', input: '', output: 'A neural network is like a brain...' },
    { instruction: 'Translate to French: Hello world', input: '', output: 'Bonjour le monde' },
  ],
}

export const MOCK_CHAT_MESSAGES = [
  {
    role: 'user' as const,
    content: 'Can you explain what QLoRA is?',
  },
  {
    role: 'assistant' as const,
    content: `**QLoRA** (Quantized Low-Rank Adaptation) is a fine-tuning technique that combines two approaches:

1. **4-bit quantization** — The base model weights are loaded in 4-bit precision (NF4), reducing VRAM by ~4×
2. **LoRA adapters** — Instead of updating all model weights, small trainable adapter matrices are added to specific layers

This lets you fine-tune large models (7B, 13B, 70B) on consumer GPUs that couldn't otherwise fit the full model in VRAM.

**Key benefits:**
- 7B model fine-tuning on 8GB VRAM (vs ~28GB for full fine-tune)
- Near-identical results to full fine-tuning
- Only the small adapter weights are saved (~50-200MB vs full model)`,
  },
]
