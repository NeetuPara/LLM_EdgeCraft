/**
 * RunDetailScreen — /run/:id
 *
 * Each stage component is a pixel-perfect read-only replica of the
 * corresponding wizard / training / export screen.
 * Card classes, font sizes, icon sizes, grid layouts, chart options —
 * everything copied verbatim from the originals and made non-interactive.
 */
import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  CheckCircle, ArrowLeft, MessageSquare, Eye,
  Zap, Layers, Database, Clock, TrendingDown,
  Settings2, Download, Bot, User, ExternalLink,
  AlertCircle, Cpu, Activity, Link, Package,
  ChevronDown,
} from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from 'recharts'
import { motion } from 'framer-motion'
import { useLiveQuery } from 'dexie-react-hooks'
import NavBar from '@/components/NavBar'
import { trainingApi, type RunDetail } from '@/api/training-api'
import { chatDb } from '@/db/chat-db'
import type { Thread, Message } from '@/db/chat-db'
import { isMockMode } from '@/api/mock'
import { apiFetch } from '@/api/client'
import { cn } from '@/utils/cn'

// ─── Stage config ─────────────────────────────────────────────────────────────

type StageId = 'model' | 'dataset' | 'params' | 'launch' | 'training' | 'chat' | 'export'

const STAGES: { id: StageId; label: string; title: string; description: string }[] = [
  { id: 'model',    label: 'Model Selection',  title: 'Model Selection',   description: 'Base model and training approach used in this experiment.' },
  { id: 'dataset',  label: 'Data Preparation', title: 'Data Preparation',  description: 'Dataset configuration, column mapping, and system prompt.' },
  { id: 'params',   label: 'Training Config',  title: 'Training Config',   description: 'Training configuration and LoRA adapter settings.' },
  { id: 'launch',   label: 'Launch',           title: 'Launch',            description: 'Full experiment configuration at the time of launch.' },
  { id: 'training', label: 'Craft',            title: 'Craft',             description: 'Loss curves, GPU stats, and training metrics from this run.' },
  { id: 'chat',     label: 'Chat',             title: 'Chat & Inference',  description: 'Conversations using this fine-tuned model.' },
  { id: 'export',   label: 'Export',           title: 'Export Model',      description: 'Export your fine-tuned model in your preferred format.' },
]

// Training/Chat use max-w-5xl (matching their original screens); all others use max-w-3xl
const WIDE_STAGES: StageId[] = ['training', 'chat']

// ─── Helpers ─────────────────────────────────────────────────────────────────

function parseConfig(run: RunDetail | null): Record<string, unknown> {
  if (!run) return {}
  try { return JSON.parse(run.config_json ?? '{}') as Record<string, unknown> } catch { return {} }
}

const fmt = {
  duration: (s?: number) => {
    if (!s) return '—'
    if (s < 60)   return `${Math.round(s)}s`
    if (s < 3600) return `${Math.round(s / 60)}m`
    return `${(s / 3600).toFixed(1)}h`
  },
  date: (iso?: string) => iso
    ? new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' })
    : '—',
  ago: (ts: number) => {
    const d = Date.now() - ts
    if (d < 60_000)     return 'just now'
    if (d < 3_600_000)  return `${Math.floor(d / 60_000)}m ago`
    if (d < 86_400_000) return `${Math.floor(d / 3_600_000)}h ago`
    return `${Math.floor(d / 86_400_000)}d ago`
  },
}

// Extract the user-given fine-tuned model name.
// Priority:
//  1. output_dir (actual saved folder path, set after training completes)
//  2. config_json.output_dir (user-given name stored at launch time — works even if output_dir is null)
//  3. Base model short name as last resort
function getFineTunedName(run: RunDetail): string {
  if (run.output_dir) {
    const folder = run.output_dir.replace(/\\/g, '/').split('/').filter(Boolean).pop() ?? ''
    if (folder) {
      const clean = folder
        .replace(/_\d{8}_\d{6}$/, '')   // strip _YYYYMMDD_HHMMSS
        .replace(/_\d{10,13}$/, '')      // strip epoch timestamp
      return clean || folder
    }
  }
  // Fallback: user-given name from config_json (stored at launch, before output_dir is written)
  try {
    const c = JSON.parse(run.config_json ?? '{}') as Record<string, unknown>
    if (c.output_dir) {
      const name = String(c.output_dir).replace(/\\/g, '/').split('/').filter(Boolean).pop() ?? ''
      if (name) return name
    }
  } catch { /* ignore */ }
  return run.model_name.split('/').pop() ?? run.model_name
}

function getDatasetName(name?: string): string {
  if (!name) return '—'
  return name.replace(/\\/g, '/').split('/').filter(Boolean).pop() ?? name
}

function threadMatchesRun(t: Thread, run: RunDetail) {
  const m = t.modelName ?? ''
  if (!m) return false
  if (m === run.model_name) return true
  if (run.output_dir && m.startsWith(run.output_dir)) return true
  if (run.output_dir) {
    const tail = run.output_dir.split(/[\\/]/).filter(Boolean).pop() ?? ''
    if (tail && m.includes(tail)) return true
  }
  // Also match by the clean fine-tuned name (in case user loaded via short name)
  const fineName = getFineTunedName(run)
  if (fineName && m.includes(fineName)) return true
  return false
}

// ─── Stage: Model ─────────────────────────────────────────────────────────────
// Exact replica of ModelSelectionScreen — cards use identical classes

function StageModel({ run, cfg }: { run: RunDetail; cfg: Record<string, unknown> }) {
  const modelType    = String(cfg.model_type ?? cfg.modelType ?? 'text')
  const trainingType = String(cfg.training_type ?? 'qlora').toLowerCase()
  // is_dataset_image is the authoritative flag (set by frontend when modelType==='vision')
  const isVision = !!cfg.is_dataset_image || modelType === 'vision'
  const isQLoRA  = trainingType.includes('qlora') || trainingType.includes('4bit') || cfg.load_in_4bit === true
  const isLoRA   = !isQLoRA && (trainingType.includes('lora') || trainingType.includes('full'))

  // Read-only badge: dimmed non-selected, but NOT opacity-40 — same as wizard hover state
  const typeCard = (active: boolean, label: string, Icon: React.ElementType, desc: string, tags: string[]) => (
    <div className={cn(
      'flex items-start gap-4 p-5 rounded-xl border text-left transition-all duration-200',
      active
        ? 'bg-cap-cyan/10 border-cap-cyan/40 ring-1 ring-cap-cyan/20'
        : 'bg-slate-800/40 border-white/[0.08]',
    )}>
      <div className={cn('p-2.5 rounded-xl shrink-0 mt-0.5', active ? 'bg-cap-cyan/15' : 'bg-slate-700/50')}>
        <Icon size={18} className={active ? 'text-cap-cyan' : 'text-slate-400'} />
      </div>
      <div className="flex-1 min-w-0">
        <p className={cn('font-semibold text-sm mb-1', active ? 'text-cap-cyan' : 'text-slate-200')}>{label}</p>
        <p className="text-xs text-slate-500 mb-2 leading-relaxed">{desc}</p>
        <div className="flex flex-wrap gap-1">
          {tags.map(t => (
            <span key={t} className="text-[10px] text-slate-600 bg-slate-800/80 border border-white/[0.06] px-1.5 py-0.5 rounded-full">{t}</span>
          ))}
        </div>
      </div>
    </div>
  )

  const methodCard = (active: boolean, label: string, badge: string, Icon: React.ElementType, color: string, desc: string, vram: string, recommended?: boolean) => (
    <div className={cn(
      'flex items-start gap-4 p-5 rounded-xl border text-left transition-all duration-200',
      active
        ? `bg-${color}/10 border-${color}/40 ring-1 ring-${color}/20`
        : 'bg-slate-800/40 border-white/[0.08]',
    )}>
      <div className={cn('p-2.5 rounded-xl shrink-0 mt-0.5', active ? `bg-${color}/15` : 'bg-slate-700/50')}>
        <Icon size={18} className={active ? `text-${color}` : 'text-slate-400'} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap mb-1">
          <span className={cn('font-semibold text-sm', active ? `text-${color}` : 'text-slate-200')}>{label}</span>
          <span className="text-[10px] text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded">{badge}</span>
          {recommended && active && (
            <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-1.5 py-0.5 rounded-full">Recommended</span>
          )}
        </div>
        <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
        <p className="text-xs text-slate-600 mt-1.5 flex items-center gap-1"><Cpu size={10} /> {vram} VRAM</p>
      </div>
    </div>
  )

  return (
    <div className="space-y-5">
      {/* Model Type — glass-card p-5, same as ModelSelectionScreen */}
      <div className="glass-card p-5">
        <div className="flex items-center gap-2 mb-4">
          <h2 className="text-sm font-semibold text-slate-300">Model Type</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {typeCard(!isVision, 'Text',   MessageSquare,
            'Fine-tune language models for conversations, writing, coding, and Q&A.',
            ['Chatbots', 'Code assistant', 'Summarization', 'Q&A', 'Translation'])}
          {typeCard(isVision,  'Vision', Eye,
            'Fine-tune multimodal models that understand both images and text.',
            ['Image captioning', 'Visual Q&A', 'Document OCR', 'Chart understanding'])}
        </div>
      </div>

      {/* Model list */}
      <div className="glass-card p-5">
        <div className="flex items-center gap-2 mb-4">
          <h2 className="text-sm font-semibold text-slate-300">{isVision ? 'Vision' : 'Text'} Models</h2>
        </div>
        {/* Selected model row — same style as selected model list item */}
        <div className="space-y-1.5">
          <div className="w-full flex items-center gap-3 px-4 py-3 rounded-xl border text-sm bg-cap-cyan/10 border-cap-cyan/30">
            <CheckCircle size={14} className="text-cap-cyan shrink-0" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5">
                <span className="font-medium text-cap-cyan truncate">{getFineTunedName(run)}</span>
              </div>
              <p className="text-[10px] text-slate-500 mt-0.5">{run.model_name}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Training Method — glass-card p-5 */}
      <div className="glass-card p-5">
        <div className="flex items-center gap-2 mb-4">
          <h2 className="text-sm font-semibold text-slate-300">Training Method</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {methodCard(isQLoRA,            'QLoRA',       '4-bit',  Zap,    'cap-cyan',    '4-bit quantized base model + LoRA adapters. Lowest VRAM — best starting point.', '~4–12 GB', true)}
          {methodCard(isLoRA && !isQLoRA, 'LoRA',        '16-bit', Layers, 'indigo-400',  '16-bit base model + LoRA adapters. Higher quality, more VRAM.',                  '~10–24 GB')}
        </div>
      </div>
    </div>
  )
}

// ─── Stage: Dataset ───────────────────────────────────────────────────────────

function StageDataset({ run, cfg }: { run: RunDetail; cfg: Record<string, unknown> }) {
  const isLocal = !!cfg.local_datasets
  const mapping = cfg.custom_format_mapping as Record<string, string> | undefined
  const isVlm   = !!cfg.is_dataset_image

  const sourceCard = (active: boolean, label: string, desc: string) => (
    <div className={cn(
      'flex items-start gap-4 p-5 rounded-xl border text-left',
      active ? 'bg-cap-cyan/10 border-cap-cyan/40 ring-1 ring-cap-cyan/20' : 'bg-slate-800/40 border-white/[0.08]',
    )}>
      <div className={cn('p-2.5 rounded-xl shrink-0 mt-0.5', active ? 'bg-cap-cyan/15' : 'bg-slate-700/50')}>
        <Database size={18} className={active ? 'text-cap-cyan' : 'text-slate-400'} />
      </div>
      <div>
        <p className={cn('font-semibold text-sm mb-1', active ? 'text-cap-cyan' : 'text-slate-200')}>{label}</p>
        <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
      </div>
    </div>
  )

  return (
    <div className="space-y-5">
      {/* Source */}
      <div className="glass-card p-5">
        <h2 className="text-sm font-semibold text-slate-300 mb-4">Dataset Source</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {sourceCard(!isLocal, 'HuggingFace Hub', 'Load dataset directly from the HuggingFace Hub.')}
          {sourceCard(isLocal,  'Local File',       'Uploaded local dataset file (JSON, CSV, Parquet).')}
        </div>
      </div>

      {/* Dataset info */}
      <div className="glass-card p-5">
        <h2 className="text-sm font-semibold text-slate-300 mb-4">Dataset Configuration</h2>
        <div className="grid grid-cols-2 gap-4">
          <div className="col-span-2">
            <label className="text-xs text-slate-500 mb-1.5 block">Dataset</label>
            <div className="glass-input py-2.5 text-sm text-slate-200">{getDatasetName(run.dataset_name)}</div>
          </div>
          {cfg.format_type && (
            <div>
              <label className="text-xs text-slate-500 mb-1.5 block">Format Type</label>
              <div className="glass-input py-2.5 text-sm text-slate-200">{cfg.format_type as string}</div>
            </div>
          )}
          {cfg.train_split && (
            <div>
              <label className="text-xs text-slate-500 mb-1.5 block">Train Split</label>
              <div className="glass-input py-2.5 text-sm text-slate-200">{cfg.train_split as string}</div>
            </div>
          )}
          {cfg.eval_split && (
            <div>
              <label className="text-xs text-slate-500 mb-1.5 block">Eval Split</label>
              <div className="glass-input py-2.5 text-sm text-slate-200">{cfg.eval_split as string}</div>
            </div>
          )}
        </div>
      </div>

      {/* Column mapping */}
      {mapping && Object.keys(mapping).length > 0 && (
        <div className="glass-card p-5">
          <h2 className="text-sm font-semibold text-slate-300 mb-4">Column Mapping</h2>
          <div className="space-y-2">
            {Object.entries(mapping).map(([col, role]) => (
              <div key={col} className="flex items-center gap-3 px-4 py-3 rounded-xl bg-slate-800/40 border border-white/[0.08]">
                <span className="text-sm text-slate-300 font-mono flex-1">{col}</span>
                <span className="text-xs text-slate-600">→</span>
                <span className={cn(
                  'text-xs font-semibold px-2.5 py-1 rounded-full border',
                  role === 'input'  ? 'bg-blue-500/10 text-blue-400 border-blue-500/20' :
                  role === 'output' ? 'bg-violet-500/10 text-violet-400 border-violet-500/20' :
                  role === 'image'  ? 'bg-amber-500/10 text-amber-400 border-amber-500/20' :
                                     'bg-slate-700 text-slate-400 border-white/10',
                )}>{role}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* System prompt */}
      {cfg.system_prompt && (
        <div className="glass-card p-5">
          <h2 className="text-sm font-semibold text-slate-300 mb-3">System Prompt</h2>
          <div className="glass-input py-3 text-sm text-slate-200 whitespace-pre-wrap leading-relaxed min-h-[80px]">
            {cfg.system_prompt as string}
          </div>
        </div>
      )}

      {/* VLM fields */}
      {isVlm && (
        <div className="glass-card p-5">
          <h2 className="text-sm font-semibold text-slate-300 mb-4">Vision Dataset</h2>
          <div className="grid grid-cols-2 gap-4">
            {cfg.image_column && (
              <div>
                <label className="text-xs text-slate-500 mb-1.5 block">Image Column</label>
                <div className="glass-input py-2.5 text-sm text-slate-200">{cfg.image_column as string}</div>
              </div>
            )}
            {cfg.dataset_base_dir && (
              <div className="col-span-2">
                <label className="text-xs text-slate-500 mb-1.5 block">Dataset Base Dir</label>
                <div className="glass-input py-2.5 text-xs text-slate-200 font-mono break-all">{cfg.dataset_base_dir as string}</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Stage: Params ────────────────────────────────────────────────────────────
// Read-only toggle + field display matching HyperparamsScreen visual style

function ReadToggle({ label, checked }: { label: string; checked: boolean }) {
  return (
    <div className="flex items-center justify-between py-2.5">
      <span className="text-sm text-slate-300">{label}</span>
      <div className={cn(
        'w-8 h-4 rounded-full relative transition-colors',
        checked ? 'bg-cap-cyan' : 'bg-slate-700',
      )}>
        <div className={cn(
          'absolute top-0.5 w-3 h-3 rounded-full bg-white shadow-sm transition-transform',
          checked ? 'translate-x-4' : 'translate-x-0.5',
        )} />
      </div>
    </div>
  )
}

function ReadField({ label, value, mono = false }: { label: string; value?: string | number | null; mono?: boolean }) {
  if (value === null || value === undefined || value === '') return null
  return (
    <div>
      <label className="flex items-center gap-1.5 text-xs text-slate-500 mb-1.5">{label}</label>
      <div className={cn('glass-input py-2.5 text-sm text-slate-200', mono && 'font-mono text-xs')}>{String(value)}</div>
    </div>
  )
}

function ReadSection({ title, children, defaultOpen = true }: {
  title: string; children: React.ReactNode; defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="glass-card p-0 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2.5 px-5 py-4 text-sm font-semibold text-slate-300 hover:bg-white/[0.02] transition-colors"
      >
        <Activity size={15} className="text-slate-500" />
        {title}
        <ChevronDown size={14} className={cn('ml-auto text-slate-600 transition-transform', open && 'rotate-180')} />
      </button>
      {open && (
        <div className="px-5 pb-5 border-t border-white/[0.06]">
          <div className="pt-4">{children}</div>
        </div>
      )}
    </div>
  )
}

function StageParams({ cfg }: { cfg: Record<string, unknown> }) {
  const g = (k: string) => cfg[k] as string | number | null | undefined
  const b = (k: string) => Boolean(cfg[k])
  const str = (k: string) => (g(k) !== null && g(k) !== undefined) ? String(g(k)) : undefined
  const isVlm = !!cfg.is_dataset_image

  return (
    <div className="space-y-4">
      <ReadSection title="Essential">
        <div className="grid grid-cols-2 gap-4">
          <ReadField label="Epochs"         value={g('num_epochs')} />
          <ReadField label="Max Steps"      value={g('max_steps')} />
          <ReadField label="Learning Rate"  value={g('learning_rate')} />
          <ReadField label="Batch Size"     value={g('batch_size')} />
          <ReadField label="Max Seq Length" value={g('max_seq_length')} suffix="tokens" />
          <ReadField label="LR Scheduler"   value={str('lr_scheduler_type')} />
          <div className="col-span-2 pt-3 border-t border-white/[0.06] space-y-1">
            <ReadToggle label="RSLoRA (Rank-Stabilized)"    checked={b('use_rslora')} />
            <ReadToggle label="Train on Completions Only"   checked={b('train_on_completions')} />
          </div>
        </div>
      </ReadSection>

      <ReadSection title="LoRA" defaultOpen={false}>
        <div className="grid grid-cols-2 gap-4">
          <ReadField label="LoRA Rank (r)"   value={g('lora_r')} />
          <ReadField label="LoRA Alpha"      value={g('lora_alpha')} />
          <ReadField label="LoRA Dropout"    value={g('lora_dropout')} />
          <ReadField label="LoftQ"           value={b('use_loftq') ? 'Yes' : 'No'} />
          {cfg.target_modules && <div className="col-span-2"><ReadField label="Target Modules" value={str('target_modules')} mono /></div>}
        </div>
      </ReadSection>

      <ReadSection title="Advanced" defaultOpen={false}>
        <div className="grid grid-cols-2 gap-4">
          <ReadField label="Grad Accumulation"   value={g('gradient_accumulation_steps')} />
          <ReadField label="Warmup Steps"        value={g('warmup_steps')} />
          <ReadField label="Weight Decay"        value={g('weight_decay')} />
          <ReadField label="LoRA Dropout"        value={g('lora_dropout')} />
          <div className="col-span-2">
            <label className="text-xs text-slate-500 mb-2 block">Checkpoint Saving</label>
            <div className="grid grid-cols-3 gap-2">
              {(['no', 'epoch', 'best'] as const).map(v => {
                const labels = { no: 'Last Only', epoch: 'Every Epoch', best: 'Best Epoch' }
                const descs  = { no: 'Save once at end', epoch: 'Save after each epoch', best: 'Lowest eval loss only' }
                const active = (cfg.save_strategy ?? 'no') === v
                return (
                  <div key={v} className={cn(
                    'flex flex-col items-center gap-0.5 px-3 py-2.5 rounded-xl border text-xs',
                    active ? 'bg-cap-cyan/10 border-cap-cyan/30 text-cap-cyan' : 'bg-slate-800/30 border-white/[0.06] text-slate-400',
                  )}>
                    <span className="font-medium">{labels[v]}</span>
                    <span className="text-[10px] opacity-60">{descs[v]}</span>
                  </div>
                )
              })}
            </div>
          </div>
          <ReadField label="Optimizer"           value={str('optim') ?? str('optimizer')} />
          <div className="col-span-2 pt-2 border-t border-white/[0.06]">
            <ReadToggle label="Sample Packing" checked={b('packing')} />
          </div>
        </div>
      </ReadSection>

      {isVlm && (
        <ReadSection title="Vision Layers">
          <div className="space-y-1">
            <ReadToggle label="Fine-tune Vision Encoder"    checked={b('finetune_vision_layers')} />
            <ReadToggle label="Fine-tune Language Layers"   checked={b('finetune_language_layers')} />
            <ReadToggle label="Fine-tune Attention Modules" checked={b('finetune_attention_modules')} />
            <ReadToggle label="Fine-tune MLP Modules"       checked={b('finetune_mlp_modules')} />
          </div>
        </ReadSection>
      )}
    </div>
  )
}

// ─── Stage: Launch ────────────────────────────────────────────────────────────

function StageLaunch({ run, cfg }: { run: RunDetail; cfg: Record<string, unknown> }) {
  const g = (k: string) => cfg[k] as string | number | boolean | undefined
  const isVlm   = !!cfg.is_dataset_image
  const effBatch = Number(g('batch_size') ?? 2) * Number(g('gradient_accumulation_steps') ?? 4)

  const rows = [
    { label: 'Fine-tuned Name',  value: getFineTunedName(run) },
    { label: 'Base Model',       value: run.model_name },
    { label: 'Type',            value: isVlm ? 'Vision (VLM)' : 'Text' },
    { label: 'Method',          value: String(g('training_type') ?? 'QLoRA') },
    { label: 'Dataset',         value: getDatasetName(run.dataset_name) },
    { label: 'Format',          value: g('format_type') },
    { label: 'Epochs',          value: g('num_epochs') },
    { label: 'Learning Rate',   value: g('learning_rate') },
    { label: 'Batch Size',      value: g('batch_size') },
    { label: 'Effective Batch', value: effBatch },
    { label: 'Max Seq Length',  value: g('max_seq_length') },
    { label: 'LoRA r / α',      value: `${g('lora_r')} / ${g('lora_alpha')}` },
    { label: 'Optimizer',       value: (g('optim') || g('optimizer')) as string },
    { label: 'LR Scheduler',    value: g('lr_scheduler_type') },
    { label: 'RSLoRA',          value: g('use_rslora') },
    { label: 'Packing',         value: g('packing') },
    { label: 'Train on Compl.', value: g('train_on_completions') },
  ].filter(r => r.value !== null && r.value !== undefined && r.value !== '')

  return (
    <div className="space-y-5">
      <div className="glass-card p-0 overflow-hidden">
        <div className="flex items-center gap-2.5 px-5 py-4 border-b border-white/[0.06]">
          <CheckCircle size={16} className="text-emerald-400" />
          <span className="font-semibold text-slate-200">Experiment Configuration</span>
          <span className="ml-auto text-sm font-semibold text-emerald-400">Completed</span>
        </div>
        <div className="divide-y divide-white/[0.04]">
          {rows.map(({ label, value }) => (
            <div key={label} className="flex items-center justify-between px-5 py-3">
              <span className="text-sm text-slate-500">{label}</span>
              <span className="text-sm text-slate-200 font-mono text-right max-w-[55%] truncate">
                {typeof value === 'boolean' ? (value ? 'Yes' : 'No') : String(value)}
              </span>
            </div>
          ))}
        </div>
      </div>
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'Started',  value: fmt.date(run.started_at) },
          { label: 'Ended',    value: fmt.date(run.ended_at) },
          { label: 'Duration', value: fmt.duration(run.duration_seconds) },
        ].map(({ label, value }) => (
          <div key={label} className="glass-card py-4 text-center">
            <p className="text-xs text-slate-500 uppercase tracking-wider mb-1.5">{label}</p>
            <p className="text-sm font-semibold text-slate-200">{value}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Stage: Training ─────────────────────────────────────────────────────────
// Exact replica of TrainingScreen (chart options, strokeWidth=2, YAxis width=50,
// axis labels, log console, completed banner)

function ChartTooltip({ active, payload, label, valueLabel, decimals = 4 }: {
  active?: boolean; payload?: Array<{ value?: number }>; label?: string | number
  valueLabel?: string; decimals?: number
}) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800/95 border border-white/10 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-500 mb-0.5">Step {label}</p>
      <p className="text-white font-medium">{valueLabel ?? ''} {Number(payload[0]?.value ?? 0).toFixed(decimals)}</p>
    </div>
  )
}

// Exact match to TrainingScreen's ChartCard
function ChartCard({
  title, data, color, valueLabel, decimals = 4, emptyText, yAxisLabel, xAxisLabel,
}: {
  title: string; data: { step: number; value: number }[]
  color: string; valueLabel?: string; decimals?: number; emptyText?: string
  yAxisLabel?: string; xAxisLabel?: string
}) {
  const hasData = data.length >= 2
  const lastVal = hasData ? Number(data[data.length - 1]?.value ?? 0) : null

  return (
    <div className="glass-card p-4 flex flex-col gap-2">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-slate-300">{title}</h3>
          {yAxisLabel && (
            <p className="text-[10px] text-slate-600 mt-0.5">
              Y: {yAxisLabel} &nbsp;·&nbsp; X: {xAxisLabel ?? 'Training Step'}
            </p>
          )}
        </div>
        {lastVal !== null && (
          <span className="text-sm font-mono shrink-0" style={{ color }}>
            {lastVal.toFixed(decimals)}
          </span>
        )}
      </div>
      <div className="h-36">
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 4, right: 8, bottom: 20, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
              <XAxis
                dataKey="step"
                stroke="#475569"
                tick={{ fontSize: 9, fill: '#475569' }}
                tickLine={false}
                axisLine={false}
                label={{ value: xAxisLabel ?? 'Step', position: 'insideBottom', offset: -10, fontSize: 9, fill: '#475569' }}
              />
              <YAxis
                stroke="#475569"
                tick={{ fontSize: 9, fill: '#475569' }}
                tickLine={false}
                axisLine={false}
                width={50}
                domain={['auto', 'auto']}
                tickFormatter={(v: number) => decimals === 6 ? v.toExponential(1) : v.toFixed(2)}
              />
              <Tooltip content={(props: unknown) => <ChartTooltip {...(props as object)} valueLabel={valueLabel} decimals={decimals} />} />
              <Line
                type="monotone"
                dataKey="value"
                stroke={color}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
                activeDot={{ r: 3, fill: color }}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full flex items-center justify-center">
            <p className="text-xs text-slate-600">{emptyText ?? 'Waiting for data...'}</p>
          </div>
        )}
      </div>
    </div>
  )
}

function StageTraining({ run }: { run: RunDetail }) {
  const m = run.metrics
  const toP = (steps: number[], vals: number[]) => steps.map((s, i) => ({ step: s, value: vals[i] }))

  const trainLoss = toP(m.loss_step_history,      m.loss_history)
  const evalLoss  = toP(m.eval_step_history,      m.eval_loss_history)
  const lr        = toP(m.lr_step_history,        m.lr_history)
  const gradNorm  = toP(m.grad_norm_step_history, m.grad_norm_history)

  return (
    <div className="space-y-5">
      {/* Header card — same as TrainingScreen's header */}
      <div className="glass-card p-5">
        <div className="flex items-start justify-between gap-4 flex-wrap mb-4">
          <div>
            <div className="flex items-center gap-2.5 mb-1">
              <CheckCircle size={16} className="text-emerald-400" />
              <span className="text-base font-bold text-slate-100">{getFineTunedName(run)}</span>
              <span className="text-sm text-slate-400 hidden md:block">{run.model_name}</span>
            </div>
            <div className="flex items-center gap-3 text-xs text-slate-500 flex-wrap">
              <span className="text-emerald-400 font-semibold">Completed</span>
              {run.final_loss != null && (
                <><span className="text-slate-600">·</span><span className="flex items-center gap-1"><TrendingDown size={11} /> {run.final_loss.toFixed(4)}</span></>
              )}
              <span className="text-slate-600">·</span>
              <span className="flex items-center gap-1"><Clock size={11} /> {fmt.duration(run.duration_seconds)}</span>
            </div>
          </div>
        </div>
        {/* Progress bar at 100% */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400">
              {run.final_step ? `Step ${run.final_step.toLocaleString()} / ${run.total_steps?.toLocaleString() ?? run.final_step.toLocaleString()}` : 'Complete'}
            </span>
            <span className="text-emerald-400 font-semibold">100%</span>
          </div>
          <div className="h-2.5 bg-slate-800 rounded-full overflow-hidden">
            <div className="h-full w-full rounded-full bg-emerald-400" />
          </div>
        </div>
      </div>

      {/* Charts 2×2 — exact grid from TrainingScreen */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <ChartCard title="Training Loss"  data={trainLoss} color="#00A5D9" valueLabel="loss:"    decimals={4} emptyText="No training loss data"     yAxisLabel="Loss (lower = better)"            xAxisLabel="Step" />
        <ChartCard title="Eval Loss"      data={evalLoss}  color="#22C55E" valueLabel="eval:"    decimals={4} emptyText="No eval split configured"   yAxisLabel="Loss on unseen data"              xAxisLabel="Step" />
        <ChartCard title="Learning Rate"  data={lr}        color="#F59E0B" valueLabel="lr:"      decimals={6} emptyText="No LR history"               yAxisLabel="Step size (warmup → decay)"       xAxisLabel="Step" />
        <ChartCard title="Gradient Norm"  data={gradNorm}  color="#A78BFA" valueLabel="grad norm:" decimals={4} emptyText="No grad norm data"         yAxisLabel="Model confusion (high→low = good)" xAxisLabel="Step" />
      </div>

      {/* Completed banner — exact match to TrainingScreen */}
      <div className="glass-card p-5 border border-emerald-500/20 bg-emerald-500/5">
        <div className="flex items-center gap-3">
          <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
            <CheckCircle size={20} className="text-emerald-400" />
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-slate-200 mb-0.5">Training Complete!</h3>
            <p className="text-sm text-slate-400">
              {run.output_dir
                ? <>Model saved to <code className="text-emerald-400 text-xs break-all">{run.output_dir}</code></>
                : 'Model saved to outputs directory'}
              {run.final_loss != null && ` · Final loss: ${run.final_loss.toFixed(4)}`}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

// ─── Stage: Chat ──────────────────────────────────────────────────────────────
// Exact replica of ChatScreen layout (ThreadSidebar + ChatPanel).
// Read-only: no input box; instead a "Continue in Chat" banner.

function modelDisplayName(path: string | null | undefined): string {
  if (!path) return ''
  return path.replace(/\\/g, '/').split('/').filter(Boolean).pop() ?? path
}

// Exact MessageBubble from ChatScreen (user pill + AI document style)
function ReadOnlyBubble({ msg }: { msg: Message }) {
  const isUser = msg.role === 'user'
  return (
    <div className={cn('group mb-1', isUser ? 'flex justify-end' : 'flex justify-start')}>
      {isUser ? (
        <div className="max-w-[72%] px-4 py-2.5 rounded-2xl rounded-tr-sm bg-cap-blue/25 border border-cap-blue/20 text-slate-200 text-sm space-y-2">
          {msg.imageDataUrl && (
            <img
              src={msg.imageDataUrl}
              alt="attached"
              className="max-h-48 w-auto rounded-xl border border-white/10 object-contain block"
            />
          )}
          {msg.content && <span>{msg.content}</span>}
        </div>
      ) : (
        <div className="w-full">
          <div className="flex items-center gap-2 mb-1.5">
            <div className="w-5 h-5 rounded-md bg-cap-cyan/10 border border-cap-cyan/20 flex items-center justify-center shrink-0">
              <Bot size={11} className="text-cap-cyan" />
            </div>
            <span className="text-[10px] text-cap-cyan/70 font-mono truncate">
              {msg.modelName ? modelDisplayName(msg.modelName) : 'Assistant'}
            </span>
          </div>
          <div className="pl-7 text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
            {msg.content}
          </div>
        </div>
      )}
    </div>
  )
}

function StageChat({ run }: { run: RunDetail }) {
  const navigate   = useNavigate()
  const [selected, setSelected] = useState<Thread | null>(null)
  const [msgs,     setMsgs]     = useState<Message[]>([])
  const [loading,  setLoading]  = useState(false)

  const allThreads = useLiveQuery(
    () => chatDb.threads.orderBy('updatedAt').reverse().toArray(),
    [], [] as Thread[],
  ) ?? []

  const matching = allThreads.filter(t => threadMatchesRun(t, run))
  const others   = allThreads.filter(t => !threadMatchesRun(t, run))

  useEffect(() => {
    if (!selected) { setMsgs([]); return }
    setLoading(true)
    chatDb.messages.where('threadId').equals(selected.id).sortBy('timestamp')
      .then(setMsgs).finally(() => setLoading(false))
  }, [selected])

  return (
    // Same h-screen flex structure as ChatScreen — fills the content area
    <div className="flex overflow-hidden rounded-xl border border-white/[0.06]"
      style={{ height: 'calc(100vh - 310px)', minHeight: 400 }}
    >

      {/* ── Thread sidebar — exact ThreadSidebar from ChatScreen ── */}
      <div className="w-52 flex flex-col bg-slate-900/50 border-r border-white/[0.06] shrink-0">
        {/* New Chat button — same style */}
        <div className="p-3 border-b border-white/[0.06]">
          <button
            onClick={() => navigate('/chat', { state: { autoLoadModel: run.output_dir || run.model_name } })}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl bg-cap-blue/20 border border-cap-blue/30 text-cap-cyan text-xs font-medium hover:bg-cap-blue/30 transition-colors"
          >
            <MessageSquare size={13} /> New Chat <ExternalLink size={10} className="opacity-60" />
          </button>
        </div>

        {/* Thread list */}
        <div className="flex-1 overflow-y-auto py-2">
          {matching.length > 0 && (
            <>
              <p className="text-[10px] font-semibold text-emerald-500/80 uppercase tracking-wider px-4 py-1.5">
                ✓ This Model ({matching.length})
              </p>
              {matching.map(t => (
                <div key={t.id} onClick={() => setSelected(t)}
                  className={cn(
                    'group flex items-center gap-2 px-3 py-2 mx-2 rounded-lg cursor-pointer transition-colors',
                    selected?.id === t.id
                      ? 'bg-white/[0.07] text-slate-200'
                      : 'text-slate-400 hover:bg-white/[0.03] hover:text-slate-300',
                  )}
                >
                  <MessageSquare size={12} className="shrink-0 opacity-60" />
                  <span className="text-xs flex-1 truncate">{t.title || 'New Chat'}</span>
                </div>
              ))}
            </>
          )}

          {others.length > 0 && (
            <>
              <p className="text-[10px] font-semibold text-slate-600 uppercase tracking-wider px-4 py-1.5 mt-1">
                History
              </p>
              {others.map(t => (
                <div key={t.id} onClick={() => setSelected(t)}
                  className={cn(
                    'group flex items-center gap-2 px-3 py-2 mx-2 rounded-lg cursor-pointer transition-colors',
                    selected?.id === t.id
                      ? 'bg-white/[0.07] text-slate-200'
                      : 'text-slate-500 hover:bg-white/[0.03] hover:text-slate-300 opacity-50 hover:opacity-80',
                  )}
                >
                  <MessageSquare size={12} className="shrink-0 opacity-60" />
                  <span className="text-xs flex-1 truncate">{t.title || 'New Chat'}</span>
                </div>
              ))}
            </>
          )}

          {allThreads.length === 0 && (
            <p className="text-center text-xs text-slate-600 py-8 px-3">No chats yet</p>
          )}
        </div>
      </div>

      {/* ── Message area ── */}
      <div className="flex-1 flex flex-col min-w-0">
        {selected ? (
          <>
            {/* Thread header — same as ChatPanel compare label row */}
            <div className="px-4 py-2.5 border-b border-white/[0.06] bg-slate-900/20 flex items-center justify-between shrink-0">
              <p className="text-xs font-medium text-slate-400 truncate">{selected.title}</p>
              <button
                onClick={() => navigate('/chat', { state: { selectThreadId: selected.id } })}
                className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-cap-cyan transition-colors ml-3 shrink-0"
              >
                Open in Chat <ExternalLink size={10} />
              </button>
            </div>

            {/* Messages — exact px-6 py-6 space-y-4 from ChatScreen */}
            <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
              {loading ? (
                <div className="h-full flex items-center justify-center">
                  <div className="w-5 h-5 border-2 border-cap-cyan/20 border-t-cap-cyan rounded-full animate-spin" />
                </div>
              ) : msgs.length === 0 ? (
                <div className="h-full flex items-center justify-center">
                  <div className="text-center space-y-3">
                    <div className="w-12 h-12 mx-auto rounded-2xl bg-slate-800/60 border border-white/[0.08] flex items-center justify-center">
                      <Bot size={22} className="text-slate-500" />
                    </div>
                    <p className="text-slate-500 text-sm">Empty thread</p>
                  </div>
                </div>
              ) : (
                msgs.map(m => <ReadOnlyBubble key={m.id} msg={m} />)
              )}
            </div>

            {/* Read-only footer instead of input */}
            <div className="px-4 pb-4 pt-2">
              <div className="flex items-center justify-between px-4 py-2.5 rounded-2xl border bg-slate-900/80 border-white/[0.06]">
                <span className="text-xs text-slate-600 italic">Read-only</span>
                <button
                  onClick={() => navigate('/chat', { state: { selectThreadId: selected.id } })}
                  className="flex items-center gap-1.5 text-xs text-cap-cyan hover:opacity-80 transition-opacity"
                >
                  Continue in Chat <ExternalLink size={11} />
                </button>
              </div>
            </div>
          </>
        ) : (
          // Empty state — same as ChatScreen
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center space-y-3">
              <div className="w-12 h-12 mx-auto rounded-2xl bg-slate-800/60 border border-white/[0.08] flex items-center justify-center">
                <Bot size={22} className="text-slate-500" />
              </div>
              <p className="text-slate-500 text-sm">
                {matching.length > 0
                  ? 'Select a thread to review'
                  : `No chats for ${getFineTunedName(run)}`}
              </p>
              {matching.length === 0 && (
                <button
                  onClick={() => navigate('/chat', { state: { autoLoadModel: run.output_dir || run.model_name } })}
                  className="flex items-center gap-2 mx-auto text-xs text-cap-cyan hover:opacity-80 transition-opacity"
                >
                  <MessageSquare size={12} /> Start chatting
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Stage: Export ────────────────────────────────────────────────────────────
// Matches ExportScreen exactly — same method cards, same quant options, functional

type ExportMethod = 'lora' | 'merged_16bit' | 'gguf'
type ExportStatus = 'idle' | 'loading' | 'exporting' | 'done' | 'error'

const EXPORT_METHODS: { id: ExportMethod; label: string; desc: string; size: string; icon: React.ElementType; color: string }[] = [
  { id: 'lora',        label: 'LoRA Adapter', icon: Link,    color: 'cap-cyan',   desc: 'Save adapter weights only. Requires the base model to run.',         size: '~50–200 MB' },
  { id: 'merged_16bit', label: 'Merged Model', icon: Layers,  color: 'indigo-400', desc: 'Merge LoRA into the base model (bfloat16). Standalone, ready to deploy.', size: 'Same as base model' },
  { id: 'gguf',        label: 'GGUF',         icon: Package, color: 'amber-400',  desc: 'Quantized format for Ollama, LM Studio, llama.cpp.',                size: 'Variable by quant level' },
]

const QUANT_OPTIONS = [
  { id: 'q4_k_m', label: 'Q4_K_M', description: 'Recommended — best balance', recommended: true },
  { id: 'q5_k_m', label: 'Q5_K_M', description: 'Higher quality, larger size' },
  { id: 'q8_0',   label: 'Q8_0',   description: 'Near-lossless, largest' },
  { id: 'f16',    label: 'F16',    description: 'No quantization — float16' },
  { id: 'q3_k_m', label: 'Q3_K_M', description: 'Smaller, lower quality' },
  { id: 'q2_k',   label: 'Q2_K',   description: 'Smallest, lowest quality' },
]

function StageExport({ run }: { run: RunDetail }) {
  const [method,      setMethod]      = useState<ExportMethod>('gguf')
  const [quantLevels, setQuantLevels] = useState<string[]>(['q4_k_m'])
  const [localPath,   setLocalPath]   = useState('')
  const [status,      setStatus]      = useState<ExportStatus>('idle')
  const [progress,    setProgress]    = useState(0)
  const [logLines,    setLogLines]    = useState<string[]>([])

  const addLog = (msg: string) => setLogLines(prev => [...prev, msg])

  const handleExport = async () => {
    if (status === 'exporting' || status === 'loading') return
    if (!run.output_dir) { addLog('No output directory found for this run.'); return }

    setStatus('loading')
    setLogLines([])
    setProgress(0)

    try {
      // Load checkpoint first
      addLog(`Loading checkpoint: ${run.output_dir}`)
      await apiFetch('/api/export/load-checkpoint', { method: 'POST', body: JSON.stringify({ checkpoint_path: run.output_dir }) })
      setProgress(10)
      setStatus('exporting')

      const saveDir = localPath.trim() || undefined

      if (method === 'lora') {
        addLog('Saving LoRA adapter weights...')
        const res = await apiFetch<{ message: string; save_directory: string }>('/api/export/export/lora', {
          method: 'POST', body: JSON.stringify({ save_directory: saveDir }),
        })
        setProgress(100); addLog(res.message); addLog(`Saved to: ${res.save_directory}`)
      } else if (method === 'merged_16bit') {
        addLog('Merging LoRA into base model (fp16)...')
        const res = await apiFetch<{ message: string; save_directory: string }>('/api/export/export/merged', {
          method: 'POST', body: JSON.stringify({ save_directory: saveDir, format_type: '16-bit (FP16)' }),
        })
        setProgress(100); addLog(res.message); addLog(`Saved to: ${res.save_directory}`)
      } else {
        for (let i = 0; i < quantLevels.length; i++) {
          const quant = quantLevels[i]
          addLog(`Exporting GGUF (${quant.toUpperCase()})... [${i + 1}/${quantLevels.length}]`)
          const res = await apiFetch<{ message: string; save_directory: string }>('/api/export/export/gguf', {
            method: 'POST', body: JSON.stringify({ save_directory: saveDir, quantization_method: quant.toUpperCase() }),
          })
          setProgress(15 + Math.round(75 * (i + 1) / quantLevels.length))
          addLog(res.message); addLog(`Saved: ${res.save_directory}`)
        }
        setProgress(100)
      }

      addLog('✓ Export complete!')
      setStatus('done')
    } catch (e: unknown) {
      addLog(`Error: ${(e as Error)?.message ?? 'Export failed'}`)
      setStatus('error'); setProgress(0)
    }
  }

  const canExport = run.output_dir && !isMockMode() && (method !== 'gguf' || quantLevels.length > 0)

  return (
    <div className="space-y-5">
      {/* Source — same as ExportScreen */}
      <div className="glass-card p-5 space-y-4">
        <h2 className="text-sm font-semibold text-slate-300">Source Checkpoint</h2>
        {run.output_dir ? (
          <div className="glass-input py-2.5 text-sm text-slate-200 font-mono truncate">{run.output_dir}</div>
        ) : (
          <p className="text-sm text-slate-500">No output directory found for this run.</p>
        )}
        {run.output_dir && (
          <div className="flex items-center gap-3 text-xs text-slate-500 flex-wrap">
            <span>Base: <span className="text-slate-300">{run.model_name.split('/').pop()}</span></span>
            {run.final_loss != null && <span>Loss: <span className="text-slate-300">{run.final_loss.toFixed(4)}</span></span>}
            <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">LoRA</span>
          </div>
        )}
      </div>

      {/* Export method — exact cards from ExportScreen */}
      <div className="glass-card p-5">
        <h2 className="text-sm font-semibold text-slate-300 mb-4">Export Method</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {EXPORT_METHODS.map(({ id, label, desc, size, icon: Icon, color }) => (
            <button key={id} onClick={() => setMethod(id)}
              className={cn(
                'flex flex-col gap-3 p-4 rounded-xl border text-left transition-all',
                method === id
                  ? `bg-${color}/10 border-${color}/40 ring-1 ring-${color}/20`
                  : 'bg-slate-800/40 border-white/[0.08] hover:border-white/20',
              )}
            >
              <div className={cn('p-2.5 rounded-xl w-fit', method === id ? `bg-${color}/15` : 'bg-slate-700/50')}>
                <Icon size={18} className={method === id ? `text-${color}` : 'text-slate-400'} />
              </div>
              <div>
                <p className={cn('font-semibold text-sm', method === id ? `text-${color}` : 'text-slate-200')}>{label}</p>
                <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{desc}</p>
                <p className="text-xs text-slate-600 mt-1">{size}</p>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* GGUF quant levels */}
      {method === 'gguf' && (
        <div className="glass-card p-5">
          <h2 className="text-sm font-semibold text-slate-300 mb-4">Quantization Level</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {QUANT_OPTIONS.map(q => {
              const active = quantLevels.includes(q.id)
              return (
                <button key={q.id} onClick={() => setQuantLevels(prev => prev.includes(q.id) ? prev.filter(x => x !== q.id) : [...prev, q.id])}
                  className={cn(
                    'flex flex-col gap-0.5 p-3 rounded-xl border text-left transition-all',
                    active ? 'bg-amber-500/10 border-amber-500/40' : 'bg-slate-800/40 border-white/[0.08] hover:border-white/20',
                  )}
                >
                  <div className="flex items-center gap-1.5">
                    <span className={cn('font-semibold text-xs', active ? 'text-amber-400' : 'text-slate-300')}>{q.label}</span>
                    {q.recommended && <span className="text-[9px] text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-1 py-0.5 rounded-full">Recommended</span>}
                  </div>
                  <span className="text-[10px] text-slate-500">{q.description}</span>
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Destination */}
      <div className="glass-card p-5 space-y-3">
        <h2 className="text-sm font-semibold text-slate-300">Save Location</h2>
        <div>
          <label className="text-xs text-slate-500 mb-1.5 block">Local path (leave blank for auto)</label>
          <input
            value={localPath}
            onChange={e => setLocalPath(e.target.value)}
            placeholder="D:/exports/ or leave blank"
            className="glass-input py-2.5 text-sm font-mono"
          />
        </div>
      </div>

      {/* Export button + progress */}
      {(status === 'exporting' || status === 'loading' || status === 'done' || status === 'error') && (
        <div className="glass-card p-5 space-y-3">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-slate-400">{status === 'done' ? 'Complete' : status === 'error' ? 'Failed' : 'Exporting...'}</span>
            <span className={cn('font-semibold', status === 'done' ? 'text-emerald-400' : status === 'error' ? 'text-red-400' : 'text-cap-cyan')}>{progress}%</span>
          </div>
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div className={cn('h-full rounded-full transition-all duration-500', status === 'done' ? 'bg-emerald-400' : status === 'error' ? 'bg-red-400' : 'bg-cap-cyan')}
              style={{ width: `${progress}%` }} />
          </div>
          <div className="h-32 overflow-y-auto bg-slate-950/60 rounded-xl p-3 font-mono text-xs leading-relaxed space-y-0.5">
            {logLines.map((line, i) => (
              <div key={i} className="text-slate-400">
                <span className="text-slate-700 select-none mr-2">{String(i + 1).padStart(3, ' ')} │</span>{line}
              </div>
            ))}
          </div>
        </div>
      )}

      <button
        onClick={handleExport}
        disabled={!canExport || status === 'exporting' || status === 'loading'}
        className="w-full btn-primary py-3 text-sm flex items-center justify-center gap-2 disabled:opacity-50"
      >
        {status === 'exporting' || status === 'loading' ? (
          <><div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />Exporting...</>
        ) : (
          <><Download size={15} />Export {method === 'gguf' ? `GGUF (${quantLevels.length} quant${quantLevels.length !== 1 ? 's' : ''})` : method === 'lora' ? 'LoRA Adapter' : 'Merged Model'}</>
        )}
      </button>

      {isMockMode() && (
        <p className="text-xs text-slate-600 text-center">Export is not available in demo mode.</p>
      )}
    </div>
  )
}

// ─── Main screen ──────────────────────────────────────────────────────────────

export default function RunDetailScreen() {
  const { id }   = useParams<{ id: string }>()
  const navigate  = useNavigate()

  const [run,     setRun]     = useState<RunDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState<string | null>(null)
  const [stage,   setStage]   = useState<StageId>('chat')

  const load = useCallback(async () => {
    if (!id) return
    try { setRun(await trainingApi.getRunDetail(id)) }
    catch (e: unknown) { setError((e as { message?: string })?.message ?? 'Failed to load run') }
    finally { setLoading(false) }
  }, [id])

  useEffect(() => { load() }, [load])

  const cfg       = parseConfig(run)
  const stageIdx  = STAGES.findIndex(s => s.id === stage)
  const stageInfo = STAGES[stageIdx]
  const isWide    = WIDE_STAGES.includes(stage)

  if (loading) {
    return (
      <div className="h-screen flex flex-col overflow-hidden">
        <NavBar />
        <div className="flex-1 flex items-center justify-center">
          <div className="flex flex-col items-center gap-4">
            <div className="w-10 h-10 border-2 border-cap-cyan/20 border-t-cap-cyan rounded-full animate-spin" />
            <p className="text-slate-500 text-sm">Loading experiment…</p>
          </div>
        </div>
      </div>
    )
  }

  if (error || !run) {
    return (
      <div className="h-screen flex flex-col overflow-hidden">
        <NavBar />
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="glass-card max-w-sm w-full text-center p-8">
            <AlertCircle size={32} className="text-red-400 mx-auto mb-3" />
            <p className="text-slate-300 font-semibold mb-1">Experiment not found</p>
            <p className="text-slate-500 text-sm mb-4">{error}</p>
            <button onClick={() => navigate('/dashboard')} className="btn-primary text-sm">Back to Dashboard</button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <NavBar />

      {/* ── Sticky pills bar — identical to NavBar's second row ── */}
      <div className="sticky top-0 z-40 px-6 pb-3 pt-0">
        <div className="max-w-7xl mx-auto">
          <div
            className="bg-slate-900/35 backdrop-blur-[16px] border border-white/[0.06] rounded-[10px] px-4 py-2.5 flex items-center justify-between gap-3"
            style={{ transform: 'translateZ(0)', willChange: 'transform' }}
          >
            {STAGES.map(({ id: sid, label }, idx) => {
              const isActive = stage === sid
              return (
                <button
                  key={sid}
                  onClick={() => setStage(sid)}
                  className={cn(
                    'flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all',
                    isActive
                      ? 'bg-cap-cyan/10 text-cap-cyan border border-cap-cyan/30 ring-1 ring-cap-cyan/20'
                      : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/15 cursor-pointer',
                  )}
                >
                  {isActive ? (
                    <span className="w-5 h-5 rounded-full bg-cap-cyan text-white flex items-center justify-center text-[10px] font-bold shrink-0">
                      {idx + 1}
                    </span>
                  ) : (
                    <CheckCircle size={13} className="shrink-0" />
                  )}
                  {label}
                </button>
              )
            })}
          </div>
        </div>
      </div>

      {/* ── Scrollable content ── */}
      <div className="flex-1 overflow-y-auto min-h-0">
        {/* Width: max-w-5xl for Training, max-w-3xl for all others — matching original screens */}
        <div className={cn('mx-auto px-6 py-8', isWide ? 'max-w-5xl' : 'max-w-3xl')}>

          {/* Breadcrumb — compact, above the step heading */}
          <div className="flex items-center gap-2 mb-6 text-xs text-slate-500">
            <button onClick={() => navigate('/dashboard')} className="flex items-center gap-1 hover:text-cap-cyan transition-colors">
              <ArrowLeft size={12} /> Dashboard
            </button>
            <span className="text-slate-700">·</span>
            <span className="font-semibold text-slate-300">{getFineTunedName(run)}</span>
            <span className="text-slate-600 text-[10px] font-mono hidden sm:block">({run.model_name.split('/').pop()})</span>
            <CheckCircle size={11} className="text-emerald-400" />
            {run.final_loss != null && <span className="text-cap-cyan font-semibold">Loss {run.final_loss.toFixed(4)}</span>}
            <span className="ml-auto flex items-center gap-1"><Clock size={10} />{fmt.date(run.started_at)}</span>
          </div>

          {/* Stage heading — exact WizardShell style */}
          <motion.div key={`hdr-${stage}`} initial={{ opacity: 1, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
            <p className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-2">
              Step {stageIdx + 1} of {STAGES.length}
            </p>
            <h1 className="text-2xl font-bold text-slate-100 font-display mb-1">{stageInfo?.title}</h1>
            <p className="text-slate-400 text-sm mb-8">{stageInfo?.description}</p>
          </motion.div>

          {/* Stage content */}
          <motion.div key={`body-${stage}`} initial={{ opacity: 1, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: 0.05 }}>
            {stage === 'model'    && <StageModel    run={run} cfg={cfg} />}
            {stage === 'dataset'  && <StageDataset  run={run} cfg={cfg} />}
            {stage === 'params'   && <StageParams              cfg={cfg} />}
            {stage === 'launch'   && <StageLaunch   run={run} cfg={cfg} />}
            {stage === 'training' && <StageTraining run={run} />}
            {stage === 'chat'     && <StageChat     run={run} />}
            {stage === 'export'   && <StageExport   run={run} />}
          </motion.div>

          <div className="h-8" />
        </div>
      </div>
    </div>
  )
}
