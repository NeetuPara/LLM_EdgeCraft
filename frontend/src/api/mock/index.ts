// ── Mock API layer ──
// When VITE_DEMO_MODE=true, all API calls return this mock data
// instead of hitting a real backend. Swap to real calls by setting
// VITE_DEMO_MODE=false and running the backend.

import {
  MOCK_AUTH_TOKENS, MOCK_HARDWARE, MOCK_RUNS,
  MOCK_TRAINING_STATUS, MOCK_MODELS, MOCK_LORAS,
  MOCK_CHECKPOINTS, MOCK_DATASET_CHECK, MOCK_USER,
  MOCK_DATASETS_SEARCH,
} from './data'

// Simulate realistic network delay (ms)
const delay = (ms = 300) => new Promise(res => setTimeout(res, ms))

export const isMockMode = (): boolean =>
  import.meta.env.VITE_DEMO_MODE === 'true'

// ── Auth ──
export const mockAuth = {
  status: async () => {
    await delay(100)
    return { initialized: true, must_change_password: false, email: MOCK_USER.email }
  },
  login: async (_email: string, _password: string) => {
    await delay(400)
    return MOCK_AUTH_TOKENS
  },
}

// ── System ──
export const mockSystem = {
  hardware: async () => {
    await delay(200)
    return MOCK_HARDWARE
  },
  health: async () => ({ status: 'ok', platform: 'windows', chat_only: false }),
}

// ── Training ──
export const mockTraining = {
  listRuns: async () => {
    await delay(300)
    return { runs: MOCK_RUNS, total: MOCK_RUNS.length }
  },
  getRun: async (id: string) => {
    await delay(150)
    return MOCK_RUNS.find(r => r.id === id) ?? MOCK_RUNS[0]
  },
  deleteRun: async (_id: string) => {
    await delay(200)
    return undefined
  },
  start: async () => {
    await delay(500)
    return { job_id: 'run_' + Date.now(), status: 'starting' }
  },
  stop: async () => { await delay(200) },
  reset: async () => { await delay(100) },
  status: async () => {
    await delay(150)
    return MOCK_TRAINING_STATUS
  },
  metrics: async () => {
    await delay(150)
    return MOCK_TRAINING_STATUS.metrics
  },
  hardware: async () => {
    await delay(100)
    return { gpu_utilization: 42, vram_used_mb: 5734, vram_total_mb: 24576 }
  },
}

// ── Models ──
export const mockModels = {
  list: async () => {
    await delay(200)
    return MOCK_MODELS
  },
  local: async () => {
    await delay(200)
    return MOCK_MODELS.filter(m => (m as { is_local?: boolean }).is_local)
  },
  loras: async () => {
    await delay(200)
    return MOCK_LORAS
  },
  checkpoints: async () => {
    await delay(200)
    return MOCK_CHECKPOINTS
  },
  config: async (name: string) => {
    await delay(300)
    return {
      name,
      is_vision: name.includes('VL') || name.includes('vision'),
      is_audio: false,
      is_embedding: false,
      max_position_embeddings: 8192,
      recommended_lora_r: 64,
      recommended_target_modules: 'q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
    }
  },
}

// ── Datasets ──
export const mockDatasets = {
  search: async (query: string) => {
    await delay(300)
    if (!query) return MOCK_DATASETS_SEARCH.slice(0, 6)
    const q = query.toLowerCase()
    return MOCK_DATASETS_SEARCH.filter(d =>
      d.name.toLowerCase().includes(q) || d.desc.toLowerCase().includes(q)
    ).slice(0, 8)
  },
  checkFormat: async () => {
    await delay(600)
    return MOCK_DATASET_CHECK
  },
  local: async () => {
    await delay(200)
    return [
      { name: 'alpaca_unsloth.json', path: './datasets/alpaca_unsloth.json', rows: 52002 },
      { name: 'my_custom_data.jsonl', path: './datasets/my_custom_data.jsonl', rows: 1500 },
    ]
  },
}

// ── Export ──
export const mockExport = {
  loadCheckpoint: async () => {
    await delay(800)
    return { loaded: true, is_vision: false, is_peft: true, base_model: 'unsloth/Llama-3.2-3B-Instruct' }
  },
  status: async () => ({
    loaded_checkpoint: './outputs/run_001',
    is_vision: false,
    is_peft: true,
  }),
  exportGguf: async () => { await delay(1000) },
  exportMerged: async () => { await delay(1000) },
  exportLora: async () => { await delay(500) },
}
