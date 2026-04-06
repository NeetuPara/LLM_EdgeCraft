import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  MessageSquare, Eye, Search, CheckCircle,
  Zap, Layers, Cpu, Key, X, Lock, AlertCircle,
} from 'lucide-react'
import WizardShell from './WizardShell'
import InfoTooltip from '@/components/InfoTooltip'
import { useTrainingConfigStore } from '@/stores/training-config-store'
import { isMockMode } from '@/api/mock'
import { cn } from '@/utils/cn'
import type { ModelType } from '@/types'
import { MOCK_TEXT_MODELS, MOCK_VISION_MODELS } from '@/api/mock/data'
import { apiFetch } from '@/api/client'

// ── Model type cards ──
const MODEL_TYPES = [
  { id: 'text'   as ModelType, label: 'Text',   icon: MessageSquare },
  { id: 'vision' as ModelType, label: 'Vision', icon: Eye },
]

const MODEL_TYPE_META = {
  text: {
    desc: 'Fine-tune language models for conversations, writing, coding, and Q&A.',
    examples: ['Chatbots', 'Code assistant', 'Summarization', 'Q&A', 'Translation'],
  },
  vision: {
    desc: 'Fine-tune multimodal models that understand both images and text.',
    examples: ['Image captioning', 'Visual Q&A', 'Document OCR', 'Chart understanding'],
  },
}

// ── Training method cards ──
const METHODS = [
  {
    id: 'qlora' as const, label: 'QLoRA', badge: '4-bit', recommended: true,
    desc: 'Quantized base model + LoRA adapters. Lowest VRAM — best starting point.',
    vram: '~4–12 GB', icon: Zap, color: 'cap-cyan',
  },
  {
    id: 'lora' as const, label: 'LoRA', badge: '16-bit', recommended: false,
    desc: 'Full precision base model + LoRA adapters. Higher quality, more VRAM.',
    vram: '~10–24 GB', icon: Layers, color: 'indigo-400',
  },
]

interface ModelItem {
  id: string; name: string; params?: string; size_gb?: number | null
  org?: string; year?: number; is_gated?: boolean; type?: string
  desc?: string; is_local?: boolean
}

// ── Helpers for real-mode filtering ──

/** Detect vision/multimodal models by name patterns */
function isVisionModel(name: string): boolean {
  const lower = name.toLowerCase()
  return ['-vl', '_vl', 'vl-', 'vision', 'llava', 'pali', 'smolvlm',
          'moondream', 'internvl', 'medgemma', 'blip', 'clip', 'flamingo',
          'idefics', 'cogvlm', 'minigpt', 'otter', 'mmgpt'].some(k => lower.includes(k))
}

/** Extract parameter count (in billions) from model name */
function getParamBillions(name: string): number {
  const lower = name.toLowerCase()
  // Match patterns like "7b", "1.5b", "0.5b", "500m" (= 0.5B)
  const mB = lower.match(/(\d+(?:\.\d+)?)\s*b(?!\w)/)
  if (mB) return parseFloat(mB[1])
  const mM = lower.match(/(\d+(?:\.\d+)?)\s*m(?!\w)/)
  if (mM) return parseFloat(mM[1]) / 1000
  return 99 // unknown size — exclude to be safe
}

function estimateVram(modelName: string, method: string): string {
  const lower = modelName.toLowerCase()
  let baseGb = 4
  if (lower.includes('70b') || lower.includes('72b')) baseGb = 40
  else if (lower.includes('32b')) baseGb = 20
  else if (lower.includes('13b') || lower.includes('14b')) baseGb = 9
  else if (lower.includes('7b') || lower.includes('8b')) baseGb = 5
  else if (lower.includes('3b') || lower.includes('4b')) baseGb = 3
  else if (lower.includes('1b') || lower.includes('1.5') || lower.includes('1.7') || lower.includes('1.8')) baseGb = 2
  else if (lower.includes('0.5') || lower.includes('0.6')) baseGb = 1
  const mult = method === 'qlora' ? 1.2 : 2.2
  return `~${Math.ceil(baseGb * mult)} GB`
}

export default function ModelSelectionScreen() {
  const navigate = useNavigate()
  const { modelType, modelName, trainingMethod, hfToken, patch, setHighestStep } = useTrainingConfigStore()

  const [search, setSearch] = useState('')
  const [models, setModels] = useState<ModelItem[]>([])
  const [localTab, setLocalTab] = useState<'popular' | 'local'>('popular')

  useEffect(() => { setHighestStep(0) }, [setHighestStep])

  // Demo mode: filter from curated mock lists
  useEffect(() => {
    if (!isMockMode()) return
    const source = modelType === 'vision' ? MOCK_VISION_MODELS : MOCK_TEXT_MODELS
    const filtered = search
      ? source.filter(m =>
          m.name.toLowerCase().includes(search.toLowerCase()) ||
          m.org?.toLowerCase().includes(search.toLowerCase()) ||
          m.desc?.toLowerCase().includes(search.toLowerCase())
        )
      : source
    setModels(filtered)
  }, [search, modelType])

  // Real mode: fetch popular OR local models depending on localTab
  useEffect(() => {
    if (isMockMode()) return
    const endpoint = localTab === 'local' ? '/api/models/local' : '/api/models/list'
    apiFetch<ModelItem[]>(endpoint)
      .then(list => {
        const preFiltered = list
          .filter(m => !m.name.toLowerCase().includes('-bnb-4bit'))
          .filter(m => !m.name.toLowerCase().includes('gguf'))
          .filter(m => getParamBillions(m.name) <= 4)
          .filter(m => {
            // Use explicit type field when backend provides it; fall back to name heuristic
            if (m.type === 'vision' || m.type === 'text') return modelType === m.type
            return modelType === 'vision' ? isVisionModel(m.name) : !isVisionModel(m.name)
          })
        // Dedup: drop local cache entries when a default (non-local) already covers the same model.
        // Matches by short name so google/gemma-3-4b-it (cache) deduplicates against
        // unsloth/gemma-3-4b-it (default) even though the org prefix differs.
        const defaultShortNames = new Set(
          preFiltered.filter(m => !m.is_local).map(m => m.name.split('/').pop()!.toLowerCase())
        )
        const filtered = preFiltered.filter(m =>
          !m.is_local || !defaultShortNames.has(m.name.split('/').pop()!.toLowerCase())
        )
        const searched = search
          ? filtered.filter(m => m.name.toLowerCase().includes(search.toLowerCase()))
          : filtered
        setModels(searched)
      })
      .catch(() => {})
  }, [modelType, search, localTab])

  // Reset model selection when type changes
  useEffect(() => {
    patch({ modelName: '' })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelType])

  const selectedModel = models.find(m => m.name === modelName)
  const selectedIsGated = selectedModel?.is_gated ?? false
  const needsToken = selectedIsGated && !hfToken.trim()
  const canProceed = !!modelName && !!trainingMethod && !needsToken

  return (
    <WizardShell
      step={1}
      title="Select Your Model"
      description="Choose a base model and training approach."
      onNext={() => { setHighestStep(1); navigate('/new/dataset') }}
      nextDisabled={!canProceed}
    >
      <div className="space-y-5">

        {/* ── Model Type ── */}
        <div className="glass-card p-5">
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-sm font-semibold text-slate-300">Model Type</h2>
            <InfoTooltip text="Determines which models are shown and how training data is processed." />
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {MODEL_TYPES.map(({ id, label, icon: Icon }) => {
              const isActive = modelType === id
              const meta = MODEL_TYPE_META[id as keyof typeof MODEL_TYPE_META]
              return (
                <button
                  key={id}
                  onClick={() => patch({
                    modelType: id,
                    ...(id === 'vision' ? {
                      learningRate: 2e-5,
                      finetuneVisionLayers: false,   // freeze vision encoder by default
                      finetuneLanguageLayers: true,
                      finetuneAttentionModules: true,
                      finetuneMlpModules: true,
                      packing: false,                // packing unsupported for VLM
                    } : {
                      learningRate: 2e-4,            // restore text default
                    }),
                  })}
                  className={cn(
                    'flex items-start gap-4 p-5 rounded-xl border text-left transition-all duration-200',
                    isActive
                      ? 'bg-cap-cyan/10 border-cap-cyan/40 ring-1 ring-cap-cyan/20'
                      : 'bg-slate-800/40 border-white/[0.08] hover:border-white/20',
                  )}
                >
                  <div className={cn('p-2.5 rounded-xl shrink-0 mt-0.5', isActive ? 'bg-cap-cyan/15' : 'bg-slate-700/50')}>
                    <Icon size={18} className={isActive ? 'text-cap-cyan' : 'text-slate-400'} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className={cn('font-semibold text-sm mb-1', isActive ? 'text-cap-cyan' : 'text-slate-200')}>{label}</p>
                    <p className="text-xs text-slate-500 mb-2 leading-relaxed">{meta?.desc}</p>
                    <div className="flex flex-wrap gap-1">
                      {meta?.examples.map(ex => (
                        <span key={ex} className="text-[10px] text-slate-600 bg-slate-800/80 border border-white/[0.06] px-1.5 py-0.5 rounded-full">
                          {ex}
                        </span>
                      ))}
                    </div>
                  </div>
                </button>
              )
            })}
          </div>
        </div>

        {/* ── Base Model ── */}
        <div className="glass-card p-5">
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-sm font-semibold text-slate-300">
              {modelType === 'vision' ? 'Vision' : 'Text'} Models
            </h2>
            <InfoTooltip text="Curated best-in-class models. 🔒 = requires HuggingFace account + accepting model license." />
            <div className="ml-auto flex rounded-lg border border-white/[0.08] overflow-hidden text-xs">
              {(['popular', 'local'] as const).map(tab => (
                <button
                  key={tab}
                  onClick={() => setLocalTab(tab)}
                  className={cn(
                    'px-3 py-1.5 capitalize transition-colors',
                    localTab === tab ? 'bg-cap-cyan/10 text-cap-cyan' : 'text-slate-500 hover:text-slate-300',
                  )}
                >
                  {tab}
                </button>
              ))}
            </div>
          </div>

          {/* Search */}
          <div className="relative mb-3">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder={`Search ${modelType} models or org name...`}
              className="glass-input pl-9 pr-9 py-2.5 text-sm"
            />
            {search && (
              <button onClick={() => setSearch('')} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300">
                <X size={13} />
              </button>
            )}
          </div>

          {/* Model list */}
          <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
            {models.length === 0 && (
              <p className="text-center text-slate-600 text-sm py-6">No models found</p>
            )}
            {models.map(m => {
              const selected = modelName === m.name
              const isGated = m.is_gated ?? false
              return (
                <button
                  key={m.id}
                  onClick={() => patch({ modelName: m.name })}
                  className={cn(
                    'w-full flex items-center gap-3 px-4 py-3 rounded-xl border text-sm text-left transition-all',
                    selected
                      ? 'bg-cap-cyan/10 border-cap-cyan/30'
                      : 'bg-slate-800/30 border-white/[0.06] hover:border-white/15 hover:bg-slate-800/50',
                  )}
                >
                  {/* Select indicator */}
                  {selected
                    ? <CheckCircle size={14} className="text-cap-cyan shrink-0" />
                    : <div className="w-3.5 h-3.5 rounded-full border-2 border-slate-600 shrink-0" />
                  }

                  {/* Model info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className={cn('font-medium truncate', selected ? 'text-cap-cyan' : 'text-slate-200')}>
                        {m.name.split('/').pop()}
                      </span>
                      {isGated && (
                        <span className="inline-flex items-center gap-0.5 text-[9px] text-amber-400 bg-amber-500/10 border border-amber-500/20 px-1 py-0.5 rounded shrink-0">
                          <Lock size={8} /> Gated
                        </span>
                      )}
                    </div>
                    <p className="text-[10px] text-slate-600 truncate mt-0.5">
                      {m.org && <span className="text-slate-500">{m.org}</span>}
                      {m.year && <span> · {m.year}</span>}
                      {m.desc && <span> · {m.desc}</span>}
                    </p>
                  </div>

                  {/* Size + params */}
                  <div className="text-right shrink-0 ml-2">
                    {m.params && <p className="text-xs font-semibold text-slate-400">{m.params}</p>}
                    {m.size_gb != null && <p className="text-[10px] text-slate-600">{m.size_gb}GB</p>}
                    {isGated && (
                      <p className="text-[9px] text-amber-500/80 mt-0.5 flex items-center justify-end gap-0.5">
                        <Key size={8} />HF Token
                      </p>
                    )}
                  </div>
                </button>
              )
            })}
          </div>

          {/* Custom model input */}
          <div className="mt-3 pt-3 border-t border-white/[0.06]">
            <p className="text-xs text-slate-500 mb-2">Or enter any HuggingFace model ID:</p>
            <input
              value={!models.find(m => m.name === modelName) ? modelName : ''}
              onChange={e => patch({ modelName: e.target.value })}
              placeholder={modelType === 'vision' ? 'e.g. llava-hf/llava-1.5-7b-hf' : 'e.g. mistralai/Mistral-7B-Instruct-v0.3'}
              className="glass-input py-2.5 text-sm font-mono"
            />
          </div>
        </div>

        {/* ── Training Method ── */}
        <div className="glass-card p-5">
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-sm font-semibold text-slate-300">Training Method</h2>
            <InfoTooltip text="QLoRA is recommended — 4× less VRAM with near-identical results to full precision." />
            {modelName && (
              <span className="ml-auto text-xs text-slate-500 flex items-center gap-1">
                <Cpu size={11} />
                Est. VRAM: <span className="text-cap-cyan font-semibold">{estimateVram(modelName, trainingMethod)}</span>
              </span>
            )}
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {METHODS.map(({ id, label, badge, recommended, desc, vram, icon: Icon, color }) => (
              <button
                key={id}
                onClick={() => patch({ trainingMethod: id })}
                className={cn(
                  'flex items-start gap-4 p-5 rounded-xl border text-left transition-all duration-200',
                  trainingMethod === id
                    ? `bg-${color}/10 border-${color}/40 ring-1 ring-${color}/20`
                    : 'bg-slate-800/40 border-white/[0.08] hover:border-white/20',
                )}
              >
                <div className={cn('p-2.5 rounded-xl shrink-0 mt-0.5', trainingMethod === id ? `bg-${color}/15` : 'bg-slate-700/50')}>
                  <Icon size={18} className={trainingMethod === id ? `text-${color}` : 'text-slate-400'} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap mb-1">
                    <span className={cn('font-semibold text-sm', trainingMethod === id ? `text-${color}` : 'text-slate-200')}>{label}</span>
                    <span className="text-[10px] text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded">{badge}</span>
                    {recommended && (
                      <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-1.5 py-0.5 rounded-full">
                        Recommended
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
                  <p className="text-xs text-slate-600 mt-1.5 flex items-center gap-1">
                    <Cpu size={10} /> {vram} VRAM
                  </p>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* ── HuggingFace Token — only shown when a gated model is selected ── */}
        {selectedIsGated && (
        <div className="glass-card p-5 border-amber-500/30 bg-amber-500/5">
          <div className="flex items-start gap-3">
            <Key size={15} className="text-amber-400 shrink-0 mt-0.5" />
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-3">
                <h2 className="text-sm font-semibold text-slate-300">HuggingFace Token</h2>
                <span className="text-[10px] font-bold text-amber-400 bg-amber-500/10 border border-amber-500/20 px-1.5 py-0.5 rounded-full">
                  Required for {selectedModel?.org ?? 'this model'}
                </span>
              </div>

              {/* Warning when token not yet filled */}
              {needsToken && (
                <div className="flex items-start gap-2 text-xs text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded-lg p-3 mb-3">
                  <AlertCircle size={13} className="shrink-0 mt-0.5" />
                  <span>
                    This model is <span className="font-semibold">gated</span> — requires accepting the license on HuggingFace.
                    Get your token at <span className="font-mono text-amber-200">huggingface.co/settings/tokens</span>, then accept
                    the model license on its HF page before downloading.
                  </span>
                </div>
              )}

              <input
                type="password"
                value={hfToken}
                onChange={e => patch({ hfToken: e.target.value })}
                placeholder="hf_..."
                className={cn(
                  'glass-input text-sm font-mono py-2.5',
                  needsToken && 'border-amber-500/40 focus:border-amber-400/60',
                )}
                autoFocus
              />
              <p className="text-[10px] text-slate-600 mt-1.5">
                Token is only used for downloading. Never stored on any server.
              </p>
            </div>
          </div>
        </div>
        )}

      </div>
    </WizardShell>
  )
}
