import {
  useState, useEffect, useRef, useCallback, useId,
} from 'react'
import { toast } from 'sonner'
import { useLiveQuery } from 'dexie-react-hooks'
import {
  Plus, Trash2, Settings, ChevronDown, Send, Columns2,
  MessageSquare, X, Bot, Loader, User as UserIcon,
  Zap, Layers, Upload, Eye, Type, Image as ImageIcon,
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import NavBar from '@/components/NavBar'
import {
  chatDb, createThread, deleteThread, addMessage,
  updateMessage, updateThreadTitle, getMessages,
} from '@/db/chat-db'
import type { Thread, Message } from '@/db/chat-db'
import { useChatStore, DEFAULT_PARAMS } from '@/stores/chat-store'
import { MOCK_MODELS, MOCK_LORAS } from '@/api/mock/data'
import { isMockMode } from '@/api/mock'
import { apiFetch } from '@/api/client'
import { streamDemoResponse } from '@/hooks/use-demo-stream'
import { cn } from '@/utils/cn'

/** Extract just the model/folder name from any path (handles / and \ separators) */
function modelDisplayName(path: string | null | undefined): string {
  if (!path) return ''
  return path.replace(/\\/g, '/').split('/').filter(Boolean).pop() ?? path
}

// ── Simple markdown renderer ──
function renderMarkdown(text: string): React.ReactNode[] {
  const lines = text.split('\n')
  const nodes: React.ReactNode[] = []
  let inCode = false
  let codeLang = ''
  let codeLines: string[] = []
  let key = 0

  const flushCode = () => {
    if (codeLines.length > 0) {
      nodes.push(
        <pre key={key++} className="bg-slate-950/70 border border-white/[0.08] rounded-xl p-4 my-2 overflow-x-auto">
          {codeLang && <div className="text-[10px] text-slate-500 mb-2 font-mono">{codeLang}</div>}
          <code className="text-sm text-slate-300 font-mono leading-relaxed">{codeLines.join('\n')}</code>
        </pre>
      )
      codeLines = []
      codeLang = ''
    }
  }

  for (const line of lines) {
    if (line.startsWith('```')) {
      if (!inCode) {
        inCode = true
        codeLang = line.slice(3).trim()
      } else {
        inCode = false
        flushCode()
      }
      continue
    }

    if (inCode) {
      codeLines.push(line)
      continue
    }

    if (!line.trim()) {
      nodes.push(<div key={key++} className="h-2" />)
      continue
    }

    // Bold **text**
    const parts = line.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**'))
        return <strong key={i} className="text-slate-100 font-semibold">{part.slice(2, -2)}</strong>
      if (part.startsWith('`') && part.endsWith('`'))
        return <code key={i} className="bg-slate-800/80 text-cap-cyan text-[0.85em] px-1.5 py-0.5 rounded font-mono">{part.slice(1, -1)}</code>
      return part
    })

    if (line.startsWith('# '))
      nodes.push(<h3 key={key++} className="text-lg font-bold text-slate-100 font-display mt-3 mb-1">{parts.slice(1)}</h3>)
    else if (line.startsWith('## '))
      nodes.push(<h4 key={key++} className="text-base font-semibold text-slate-200 mt-2 mb-1">{parts.slice(1)}</h4>)
    else if (line.startsWith('- ') || line.startsWith('* '))
      nodes.push(<li key={key++} className="ml-4 text-slate-300 list-disc leading-relaxed">{parts.slice(1)}</li>)
    else if (/^\d+\.\s/.test(line))
      nodes.push(<li key={key++} className="ml-4 text-slate-300 list-decimal leading-relaxed">{parts.slice(1)}</li>)
    else
      nodes.push(<p key={key++} className="text-slate-300 leading-relaxed">{parts}</p>)
  }

  if (inCode) flushCode()
  return nodes
}

// ── Message bubble ──
function MessageBubble({ msg, modelName }: { msg: Message; modelName?: string | null }) {
  const isUser = msg.role === 'user'
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18 }}
      className={cn('group mb-1', isUser ? 'flex justify-end' : 'flex justify-start')}
    >
      {isUser ? (
        /* User: compact right-aligned pill */
        <div className="max-w-[72%] px-4 py-2.5 rounded-2xl rounded-tr-sm bg-cap-blue/25 border border-cap-blue/20 text-slate-200 text-sm space-y-2">
          {msg.imageDataUrl && (
            <img src={msg.imageDataUrl} alt="attached" className="max-h-40 rounded-xl border border-white/10 object-cover" />
          )}
          {msg.content && <span>{msg.content}</span>}
        </div>
      ) : (
        /* AI: full-width document style with left accent */
        <div className="w-full">
          <div className="flex items-center gap-2 mb-1.5">
            <div className="w-5 h-5 rounded-md bg-cap-cyan/10 border border-cap-cyan/20 flex items-center justify-center shrink-0">
              <Bot size={11} className="text-cap-cyan" />
            </div>
            <span className="text-[10px] text-cap-cyan/70 font-mono truncate">
              {modelName ? modelDisplayName(modelName) : 'Assistant'}
            </span>
          </div>
          <div className="pl-7">
            {msg.streaming && msg.content === '' ? (
              <div className="flex items-center gap-1 py-1">
                {[0, 1, 2].map(i => (
                  <div key={i} className="w-1.5 h-1.5 rounded-full bg-cap-cyan/60 animate-bounce"
                    style={{ animationDelay: `${i * 0.15}s` }} />
                ))}
              </div>
            ) : (
              <div className="text-sm text-slate-200 leading-relaxed">
                {renderMarkdown(msg.content)}
                {msg.streaming && <span className="inline-block w-0.5 h-4 bg-cap-cyan ml-0.5 animate-pulse" />}
              </div>
            )}
          </div>
        </div>
      )}
    </motion.div>
  )
}

// ── Model selector popover ──
interface ModelItem { id: string; name: string; size_gb?: number | null; is_local?: boolean; path?: string }
interface LoraItem { id: string; name: string; path: string; base_model?: string | null }

function _friendlyLoraName(name: string): string {
  // "unsloth_Llama-3.2-1B-Instruct_1743619200" → "Llama-3.2-1B-Instruct (fine-tuned)"
  const parts = name.replace(/^unsloth_/, '').split('_')
  if (parts.length >= 2 && /^\d{10}$/.test(parts[parts.length - 1])) {
    parts.pop() // remove timestamp
  }
  return parts.join('-') + ' (fine-tuned)'
}

interface ExportedModel { id: string; name: string; path: string; type: 'lora' | 'merged' }

// ── Load Custom Model modal ──────────────────────────────────────────────────
// Lets the user load any local model by path:
//   • GGUF file  (path/to/model.gguf)
//   • Adapter folder (path/to/adapter/) — base model auto-detected from adapter_config.json
//   • Merged HF folder (path/to/merged/)
// ────────────────────────────────────────────────────────────────────────────

function detectModelType(path: string): { type: 'gguf' | 'adapter' | 'merged' | ''; hint: string } {
  const p = path.toLowerCase().replace(/\\/g, '/')
  if (p.endsWith('.gguf') || p.includes('.gguf'))
    return { type: 'gguf', hint: 'GGUF quantized model — loads directly via llama.cpp.' }
  if (p.includes('adapter') || p.includes('lora') || p.includes('qlora') || p.includes('peft'))
    return { type: 'adapter', hint: 'LoRA/QLoRA adapter — base model is auto-detected from adapter_config.json.' }
  if (p.includes('merged') || p.includes('full'))
    return { type: 'merged', hint: 'Merged HF model folder (base + adapter merged). Loads as a standard HF model.' }
  if (path.length > 4)
    return { type: 'adapter', hint: 'Will detect type automatically. If it contains adapter_config.json, the base model is auto-loaded.' }
  return { type: '', hint: '' }
}

function LoadCustomModelModal({
  open, onClose, onLoad,
}: { open: boolean; onClose: () => void; onLoad: (path: string) => void }) {
  const [path, setPath] = useState('')
  const [slot, setSlot] = useState<'left' | 'right'>('left')
  const detected = detectModelType(path)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) { setPath(''); setTimeout(() => inputRef.current?.focus(), 100) }
  }, [open])

  const handleLoad = () => {
    const p = path.trim()
    if (!p) return
    onLoad(p)
    onClose()
  }

  if (!open) return null

  const typeColors: Record<string, string> = {
    gguf:    'text-amber-400 bg-amber-500/10 border-amber-500/20',
    adapter: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
    merged:  'text-indigo-400 bg-indigo-500/10 border-indigo-500/20',
  }
  const typeLabels: Record<string, string> = {
    gguf: 'GGUF', adapter: 'Adapter', merged: 'Merged HF',
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.96, y: 8 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.96 }}
        transition={{ duration: 0.18 }}
        className="bg-slate-900/95 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl w-full max-w-lg p-6"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center gap-3 mb-5">
          <div className="p-2.5 rounded-xl bg-cap-cyan/10 border border-cap-cyan/20">
            <Upload size={18} className="text-cap-cyan" />
          </div>
          <div>
            <h2 className="text-base font-bold text-slate-100">Load Custom Model</h2>
            <p className="text-xs text-slate-500 mt-0.5">Load any local model — adapter, GGUF, or merged HF folder</p>
          </div>
          <button onClick={onClose} className="ml-auto p-1.5 text-slate-500 hover:text-slate-300 transition-colors">
            <X size={16} />
          </button>
        </div>

        {/* Path input */}
        <div className="mb-4">
          <label className="block text-xs text-slate-500 mb-1.5 font-medium">Model Path</label>
          <input
            ref={inputRef}
            value={path}
            onChange={e => setPath(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleLoad()}
            placeholder="D:\models\my_adapter  or  D:\models\model.gguf"
            className="w-full px-3 py-3 rounded-xl bg-slate-800/60 border border-slate-700 text-sm text-slate-200 font-mono focus:outline-none focus:border-cap-cyan/50 focus:ring-1 focus:ring-cap-cyan/20 placeholder:text-slate-600 placeholder:font-sans"
          />
        </div>

        {/* Auto-detected type badge + hint */}
        {detected.type && (
          <div className="mb-4 flex items-start gap-2.5 px-3.5 py-2.5 rounded-xl bg-slate-800/40 border border-white/[0.06]">
            <span className={cn('text-[10px] font-semibold px-2 py-0.5 rounded-full border shrink-0 mt-0.5', typeColors[detected.type])}>
              {typeLabels[detected.type]}
            </span>
            <p className="text-xs text-slate-400 leading-relaxed">{detected.hint}</p>
          </div>
        )}

        {/* Reference cards */}
        <div className="grid grid-cols-3 gap-2 mb-5">
          {[
            { type: 'gguf',    label: 'GGUF',         example: 'model.gguf',            desc: 'Fast llama.cpp inference' },
            { type: 'adapter', label: 'Adapter',       example: 'lora_output/checkpoint', desc: 'Auto-loads base model' },
            { type: 'merged',  label: 'Merged',        example: 'merged_model/',          desc: 'Full HF model folder' },
          ].map(({ type, label, example, desc }) => (
            <div key={type} className={cn('rounded-xl p-3 border', typeColors[type])}>
              <p className={cn('text-[10px] font-semibold mb-0.5', type === 'gguf' ? 'text-amber-400' : type === 'adapter' ? 'text-emerald-400' : 'text-indigo-400')}>{label}</p>
              <p className="text-[10px] font-mono text-slate-500 truncate">{example}</p>
              <p className="text-[9px] text-slate-600 mt-0.5">{desc}</p>
            </div>
          ))}
        </div>

        {/* Actions */}
        <div className="flex gap-3">
          <button onClick={onClose} className="btn-secondary flex-1 py-2.5 text-sm">Cancel</button>
          <button
            onClick={handleLoad}
            disabled={!path.trim()}
            className="btn-primary flex-1 py-2.5 text-sm flex items-center justify-center gap-2 disabled:opacity-50"
          >
            <Upload size={14} />
            Load Model
          </button>
        </div>
      </motion.div>
    </div>
  )
}

function isVisionModel(name: string): boolean {
  const lower = name.toLowerCase()
  return ['-vl', '_vl', 'vl-', 'vision', 'llava', 'pali', 'smolvlm',
          'moondream', 'internvl', 'medgemma', 'blip'].some(k => lower.includes(k))
}

function ModelSelector({
  selected, onSelect, modelType = 'text', label = 'Select Model',
}: {
  selected: string | null
  onSelect: (name: string) => void
  modelType?: 'text' | 'vision'
  label?: string
}) {
  const [open, setOpen] = useState(false)
  const [models, setModels] = useState<ModelItem[]>([])
  const [loras, setLoras] = useState<LoraItem[]>([])
  const [exported, setExported] = useState<ExportedModel[]>([])
  const [localPath, setLocalPath] = useState('')
  const ref = useRef<HTMLDivElement>(null)

  // Fetch models when popover opens
  useEffect(() => {
    if (!open) return
    if (isMockMode()) {
      setModels(MOCK_MODELS as ModelItem[])
      setLoras(MOCK_LORAS as LoraItem[])
      return
    }
    apiFetch<ModelItem[]>('/api/models/list').then(setModels).catch(() => setModels(MOCK_MODELS as ModelItem[]))
    apiFetch<LoraItem[]>('/api/models/loras').then(setLoras).catch(() => setLoras([]))
    apiFetch<ExportedModel[]>('/api/models/exported').then(setExported).catch(() => setExported([]))
  }, [open])

  // Filter by modelType
  const filteredModels = models.filter(m =>
    modelType === 'vision' ? isVisionModel(m.name) : !isVisionModel(m.name)
  )
  const filteredLoras = loras.filter(m =>
    modelType === 'vision' ? isVisionModel(m.name) : !isVisionModel(m.name)
  )
  const filteredExported = exported.filter(m =>
    modelType === 'vision' ? isVisionModel(m.name) : !isVisionModel(m.name)
  )

  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [open])

  const isSelectedLora = filteredLoras.some(l => l.name === selected || l.path === selected)
  const isSelectedExported = filteredExported.some(m => m.path === selected)
  const displayName = selected
    ? isSelectedLora
      ? _friendlyLoraName(modelDisplayName(selected) || selected)
      : modelDisplayName(selected) || selected
    : label

  const accent = isSelectedLora ? 'emerald' : isSelectedExported ? 'indigo' : 'cyan'

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          'flex items-center gap-2 px-3 py-2 rounded-xl border text-sm transition-colors max-w-[220px]',
          isSelectedLora ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
          : isSelectedExported ? 'bg-indigo-500/10 border-indigo-500/30 text-indigo-400'
          : selected ? 'bg-cap-cyan/10 border-cap-cyan/30 text-cap-cyan'
          : 'bg-slate-800/50 border-white/[0.08] text-slate-400 hover:border-cap-cyan/30 hover:text-slate-200',
        )}
      >
        {isSelectedLora ? <Zap size={13} className="text-emerald-400 shrink-0" />
         : isSelectedExported ? <Layers size={13} className="text-indigo-400 shrink-0" />
         : <Bot size={13} className={cn('shrink-0', selected ? 'text-cap-cyan' : 'text-slate-500')} />}
        <span className="truncate text-xs">{displayName}</span>
        <ChevronDown size={12} className={cn('text-slate-500 shrink-0 transition-transform', open && 'rotate-180')} />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 4, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.97 }}
            transition={{ duration: 0.15 }}
            className="absolute top-full left-0 mt-2 w-80 bg-slate-900/98 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl z-50 overflow-hidden"
          >
            {/* Header: type badge */}
            <div className="px-4 py-2.5 border-b border-white/[0.06] flex items-center gap-2">
              {modelType === 'vision'
                ? <Eye size={12} className="text-purple-400" />
                : <Type size={12} className="text-cap-cyan" />}
              <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
                {modelType === 'vision' ? 'Vision' : 'Text'} Models
              </span>
            </div>

            {/* Fine-tuned (LoRA) adapters */}
            {filteredLoras.length > 0 && (
              <>
                <div className="px-4 py-2 bg-emerald-500/5">
                  <p className="text-[10px] font-semibold text-emerald-400 uppercase tracking-wider flex items-center gap-1">
                    <Zap size={10} /> Your Fine-tuned Models
                  </p>
                </div>
                <div className="max-h-28 overflow-y-auto">
                  {filteredLoras.map(l => {
                    const isActive = selected === l.name || selected === l.path
                    return (
                      <button key={l.id} onClick={() => { onSelect(l.path || l.name); setOpen(false) }}
                        className={cn('w-full flex items-center gap-3 px-4 py-2 text-sm text-left hover:bg-white/5 transition-colors', isActive && 'bg-emerald-500/10')}>
                        <Zap size={11} className="text-emerald-400 shrink-0" />
                        <div className="flex-1 min-w-0">
                          <p className={cn('truncate font-medium text-xs', isActive ? 'text-emerald-400' : 'text-slate-200')}>
                            {_friendlyLoraName(l.name)}
                          </p>
                          {l.base_model && <p className="text-[9px] text-slate-600 truncate">base: {modelDisplayName(l.base_model)}</p>}
                        </div>
                        <span className="text-[9px] text-emerald-600 bg-emerald-500/10 px-1.5 py-0.5 rounded shrink-0">adapter</span>
                      </button>
                    )
                  })}
                </div>
              </>
            )}

            {/* Exported / Merged */}
            {filteredExported.length > 0 && (
              <>
                <div className="px-4 py-2 bg-indigo-500/5 border-t border-white/[0.04]">
                  <p className="text-[10px] font-semibold text-indigo-400 uppercase tracking-wider flex items-center gap-1">
                    <Layers size={10} /> Exported Models
                  </p>
                </div>
                <div className="max-h-28 overflow-y-auto">
                  {filteredExported.map(m => {
                    const isActive = selected === m.path
                    return (
                      <button key={m.id} onClick={() => { onSelect(m.path); setOpen(false) }}
                        className={cn('w-full flex items-center gap-3 px-4 py-2 text-sm text-left hover:bg-white/5 transition-colors', isActive && 'bg-indigo-500/10')}>
                        <Layers size={11} className="text-indigo-400 shrink-0" />
                        <span className={cn('flex-1 truncate font-medium text-xs', isActive ? 'text-indigo-400' : 'text-slate-200')}>{m.name}</span>
                        <span className="text-[9px] text-indigo-600 bg-indigo-500/10 px-1.5 py-0.5 rounded shrink-0">{m.type}</span>
                      </button>
                    )
                  })}
                </div>
              </>
            )}

            {/* Base Models */}
            <div className="px-4 py-2 border-t border-white/[0.04]">
              <p className="text-[10px] font-semibold text-slate-600 uppercase tracking-wider">Base Models</p>
            </div>
            <div className="max-h-44 overflow-y-auto">
              {filteredModels.length === 0 && (
                <p className="text-center text-xs text-slate-600 py-4">No {modelType} models found</p>
              )}
              {filteredModels.map(m => (
                <button key={m.id} onClick={() => { onSelect(m.name); setOpen(false) }}
                  className={cn('w-full flex items-center gap-3 px-4 py-2 text-xs text-left hover:bg-white/5 transition-colors', selected === m.name && 'bg-cap-cyan/5')}>
                  <div className={cn('w-1.5 h-1.5 rounded-full shrink-0', selected === m.name ? 'bg-cap-cyan' : 'bg-slate-700')} />
                  <span className={cn('flex-1 truncate', selected === m.name ? 'text-cap-cyan' : 'text-slate-300')}>{m.name.split('/').pop()}</span>
                  {m.size_gb != null && <span className="text-slate-600 shrink-0">{m.size_gb}GB</span>}
                  {m.is_local && <span className="text-[9px] text-slate-600 bg-slate-800 px-1 py-0.5 rounded">cached</span>}
                </button>
              ))}
            </div>

            {/* Load from local path */}
            <div className="px-4 py-3 border-t border-white/[0.06] bg-slate-800/30">
              <p className="text-[10px] text-slate-500 mb-1.5 flex items-center gap-1">
                <Upload size={10} /> Load from local path:
              </p>
              <div className="flex gap-1.5">
                <input
                  value={localPath}
                  onChange={e => setLocalPath(e.target.value)}
                  placeholder="D:/models/my_model"
                  className="flex-1 bg-slate-900/60 border border-white/[0.08] rounded-lg px-2 py-1.5 text-[11px] text-slate-300 font-mono focus:outline-none focus:border-cap-cyan/40"
                  onKeyDown={e => { if (e.key === 'Enter' && localPath.trim()) { onSelect(localPath.trim()); setLocalPath(''); setOpen(false) } }}
                />
                <button onClick={() => { if (localPath.trim()) { onSelect(localPath.trim()); setLocalPath(''); setOpen(false) } }}
                  className="text-[11px] text-cap-cyan px-2 py-1.5 rounded-lg border border-cap-cyan/20 hover:bg-cap-cyan/10 transition-colors whitespace-nowrap">
                  Load
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ── Settings sheet ──
function SettingsSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { params, setParams } = useChatStore()
  const configFileRef = useRef<HTMLInputElement>(null)
  const [importing, setImporting] = useState(false)
  const [lastImport, setLastImport] = useState<{ filename: string; count: number } | null>(null)

  const handleConfigUpload = async (file: File) => {
    setImporting(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await apiFetch<{ config: Record<string, unknown>; applied_count: number; filename: string }>(
        '/api/config/parse-inference', { method: 'POST', body: form }
      )
      setParams(res.config as Parameters<typeof setParams>[0])
      setLastImport({ filename: res.filename ?? file.name, count: res.applied_count })
      toast.success(`Config loaded — ${res.applied_count} values applied from "${res.filename}"`)
    } catch (err: unknown) {
      toast.error((err as { message?: string })?.message ?? 'Failed to parse config file')
    } finally {
      setImporting(false)
    }
  }

  const isGreedy = params.temperature === 0

  const slider = (
    label: string, key: keyof typeof DEFAULT_PARAMS,
    min: number, max: number, step: number,
    disabled?: boolean,
  ) => (
    <div key={key} className={cn(disabled && 'opacity-40 pointer-events-none')}>
      <div className="flex justify-between text-xs text-slate-400 mb-1.5">
        <span className="capitalize">{label}</span>
        <span className={cn('font-mono', disabled ? 'text-slate-600' : 'text-cap-cyan')}>
          {Number(params[key]).toFixed(step < 0.1 ? 2 : 1)}
        </span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step}
        value={params[key] as number}
        onChange={e => setParams({ [key]: parseFloat(e.target.value) })}
        disabled={disabled}
        className="w-full h-1.5 accent-cap-cyan rounded-full"
      />
    </div>
  )

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-black/40"
            onClick={onClose}
          />
          <motion.div
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 30, stiffness: 300 }}
            className="fixed right-0 top-0 h-full w-80 z-50 bg-slate-900/95 backdrop-blur-xl border-l border-white/[0.08] flex flex-col"
          >
            <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.06]">
              <h3 className="font-semibold text-slate-200">Inference Settings</h3>
              <button onClick={onClose} className="p-1.5 hover:bg-white/10 rounded-lg transition-colors">
                <X size={16} className="text-slate-400" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-5 space-y-5">

              {/* Greedy / Sampling toggle — sets temperature to 0 or restores 0.7 */}
              <div className="flex items-center justify-between py-2 px-3 rounded-xl bg-slate-800/50 border border-white/[0.06]">
                <div className="min-w-0">
                  <p className="text-xs font-medium text-slate-200">Decoding Mode</p>
                  <p className="text-[10px] text-slate-500 mt-0.5">
                    {isGreedy ? 'Greedy (temperature=0) — deterministic' : 'Sampling — creative, varied'}
                  </p>
                </div>
                <button
                  onClick={() => setParams({ temperature: isGreedy ? 0.7 : 0 })}
                  className={cn(
                    'relative ml-3 w-10 h-5 rounded-full transition-colors shrink-0',
                    !isGreedy ? 'bg-cap-cyan' : 'bg-slate-700',
                  )}
                >
                  <span className={cn(
                    'absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform',
                    !isGreedy ? 'translate-x-5' : 'translate-x-0.5',
                  )} />
                </button>
              </div>

              {isGreedy && (
                <p className="text-[10px] text-amber-400/80 bg-amber-400/5 border border-amber-400/20 rounded-lg px-3 py-2">
                  Greedy mode — Top P, Top K, Min P have no effect
                </p>
              )}
              {slider('Temperature', 'temperature', 0, 2, 0.05)}
              {slider('Top P', 'topP', 0, 1, 0.05, isGreedy)}
              {slider('Top K', 'topK', 1, 100, 1, isGreedy)}
              {slider('Min P', 'minP', 0, 0.5, 0.01, isGreedy)}
              {slider('Repetition Penalty', 'repetitionPenalty', 1, 2, 0.05)}

              <div>
                <label className="text-xs text-slate-400 mb-1.5 block">Max Tokens</label>
                <input
                  type="number" min={1} max={8192} step={64}
                  value={params.maxTokens}
                  onChange={e => setParams({ maxTokens: parseInt(e.target.value) })}
                  className="glass-input text-sm py-2"
                />
              </div>

              {/* System Prompt */}
              <div className="pt-1 border-t border-white/[0.06]">
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-xs text-slate-400">System Prompt</label>
                  {params.systemPrompt && (
                    <button
                      onClick={() => setParams({ systemPrompt: '' })}
                      className="text-[10px] text-slate-600 hover:text-red-400 transition-colors"
                    >
                      clear
                    </button>
                  )}
                </div>
                <textarea
                  value={params.systemPrompt}
                  onChange={e => setParams({ systemPrompt: e.target.value })}
                  placeholder="e.g. You are an expert in EU consumer protection law..."
                  rows={4}
                  className="glass-input text-xs w-full resize-none leading-relaxed"
                />
                <p className="text-[10px] text-slate-600 mt-1 leading-relaxed">
                  Pre-filled from training metadata when a fine-tuned model loads.
                  Leave empty for no system prompt.
                </p>
              </div>

              {/* Config file upload */}
              <div className="pt-1 border-t border-white/[0.06]">
                <input
                  ref={configFileRef}
                  type="file"
                  accept=".json,.yaml,.yml,.cfg,.ini,.txt"
                  className="hidden"
                  onChange={e => {
                    const file = e.target.files?.[0]
                    if (file) handleConfigUpload(file)
                    e.target.value = ''
                  }}
                />
                <button
                  onClick={() => configFileRef.current?.click()}
                  disabled={importing}
                  className="w-full flex items-center justify-center gap-2 btn-secondary text-sm py-2 disabled:opacity-50"
                >
                  {importing
                    ? <><Loader size={13} className="animate-spin" /> Importing...</>
                    : <><Upload size={13} /> Import Config File</>
                  }
                </button>
                <p className="text-[10px] text-slate-500 mt-1.5 text-center">
                  {lastImport
                    ? <span className="text-emerald-400">✓ {lastImport.count} values loaded from "{lastImport.filename}"</span>
                    : 'Upload .json · .yaml · .cfg · .txt to auto-fill all fields'
                  }
                </p>
              </div>

              <button
                onClick={() => setParams(DEFAULT_PARAMS)}
                className="w-full btn-secondary text-sm py-2"
              >
                Reset to Defaults
              </button>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

// ── Thread sidebar ──
function ThreadSidebar({
  threads, activeId, onSelect, onCreate, onDelete,
}: {
  threads: Thread[]
  activeId: string | null
  onSelect: (id: string) => void
  onCreate: () => void
  onDelete: (id: string) => void
}) {
  return (
    <div className="w-52 flex flex-col bg-slate-900/50 border-r border-white/[0.06]">
      {/* New chat */}
      <div className="p-3 border-b border-white/[0.06]">
        <button onClick={onCreate}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl bg-cap-blue/20 border border-cap-blue/30 text-cap-cyan text-xs font-medium hover:bg-cap-blue/30 transition-colors">
          <Plus size={13} /> New Chat
        </button>
      </div>

      {/* Thread list */}
      <div className="flex-1 overflow-y-auto py-2">
        <p className="text-[10px] font-semibold text-slate-600 uppercase tracking-wider px-4 py-1.5">History</p>
        {threads.length === 0 && (
          <p className="text-center text-xs text-slate-600 py-8 px-3">No chats yet</p>
        )}
        {threads.map(t => (
          <div key={t.id}
            className={cn('group flex items-center gap-2 px-3 py-2 mx-2 rounded-lg cursor-pointer transition-colors',
              activeId === t.id ? 'bg-white/[0.07] text-slate-200' : 'text-slate-500 hover:bg-white/[0.03] hover:text-slate-300')}
            onClick={() => onSelect(t.id)}>
            <MessageSquare size={12} className="shrink-0 opacity-60" />
            <span className="text-xs flex-1 truncate">{t.title}</span>
            <button onClick={(e) => { e.stopPropagation(); onDelete(t.id) }}
              className="opacity-0 group-hover:opacity-100 p-0.5 hover:text-red-400 transition-all">
              <Trash2 size={10} />
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Chat panel (single model) ──
function ChatPanel({
  messages, isStreaming, modelName, onSend, disabled, sharedInput, setSharedInput, isCompare,
  isVision = false,
}: {
  messages: Message[]
  isStreaming: boolean
  modelName: string | null
  onSend: (text: string, imageDataUrl?: string) => void
  disabled?: boolean
  sharedInput?: string
  setSharedInput?: (v: string) => void
  isCompare?: boolean
  isVision?: boolean
}) {
  const [localInput, setLocalInput] = useState('')
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const input = sharedInput !== undefined ? sharedInput : localInput
  const setInput = setSharedInput ?? setLocalInput
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputId = useId()
  const imageRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages.length, messages[messages.length - 1]?.content?.length])

  const handleSend = () => {
    const text = input.trim()
    if ((!text && !imagePreview) || isStreaming || disabled) return
    const img = imagePreview   // capture before clearing
    setInput('')
    setImagePreview(null)
    onSend(text, img ?? undefined)
  }

  return (
    <div className="flex-1 flex flex-col min-w-0">
      {/* Model label (compare mode) */}
      {isCompare && (
        <div className="px-4 py-2 border-b border-white/[0.06] bg-slate-900/20">
          <p className="text-xs font-medium text-slate-500">
            {modelName ? modelDisplayName(modelName) : 'No model selected'}
          </p>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
        {messages.length === 0 && (
          <div className="h-full flex items-center justify-center">
            <div className="text-center space-y-3">
              <div className="w-12 h-12 mx-auto rounded-2xl bg-slate-800/60 border border-white/[0.08] flex items-center justify-center">
                <Bot size={22} className="text-slate-500" />
              </div>
              <p className="text-slate-500 text-sm">
                {modelName ? `${modelDisplayName(modelName)} ready` : 'Load a model to start'}
              </p>
            </div>
          </div>
        )}
        {messages.map(msg => (
          <MessageBubble key={msg.id} msg={msg} modelName={!isCompare ? modelName : modelName} />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input — floating card style */}
      {!isCompare && (
        <div className="px-4 pb-4">
          {/* Image preview */}
          {imagePreview && (
            <div className="mb-2 relative inline-block">
              <img src={imagePreview} alt="upload" className="h-20 w-auto rounded-xl border border-white/10 object-cover" />
              <button onClick={() => setImagePreview(null)}
                className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-slate-900 border border-white/10 text-slate-400 hover:text-red-400 flex items-center justify-center">
                <X size={10} />
              </button>
            </div>
          )}
          <div className={cn(
            'flex gap-2 items-end p-3 rounded-2xl border bg-slate-900/80 backdrop-blur-sm transition-colors',
            disabled ? 'border-white/[0.04] opacity-60' : 'border-white/[0.1] focus-within:border-cap-cyan/30',
          )}>
            {/* Vision: image upload button */}
            {isVision && (
              <>
                <input ref={imageRef} type="file" accept="image/*" className="hidden"
                  onChange={e => {
                    const f = e.target.files?.[0]
                    if (f) {
                      const reader = new FileReader()
                      reader.onload = ev => setImagePreview(ev.target?.result as string)
                      reader.readAsDataURL(f)
                    }
                    e.target.value = ''
                  }}
                />
                <button onClick={() => imageRef.current?.click()}
                  disabled={disabled}
                  className="p-1.5 rounded-lg text-slate-500 hover:text-purple-400 hover:bg-purple-500/10 transition-colors shrink-0"
                  title="Upload image">
                  <ImageIcon size={16} />
                </button>
              </>
            )}
            <textarea
              id={inputId}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
              }}
              placeholder={disabled ? 'Load a model to start chatting...' : isVision ? 'Ask about the image or type a message...' : 'Ask anything...'}
              rows={1}
              disabled={disabled}
              className="flex-1 bg-transparent text-sm text-slate-200 placeholder-slate-600 resize-none focus:outline-none py-1.5"
              style={{ maxHeight: '140px', overflowY: 'auto' }}
            />
            <button
              onClick={handleSend}
              disabled={(!input.trim() && !imagePreview) || isStreaming || disabled}
              className={cn(
                'p-2.5 rounded-xl text-white transition-all shrink-0',
                (!input.trim() && !imagePreview) || isStreaming || disabled
                  ? 'bg-slate-700/50 text-slate-600 cursor-not-allowed'
                  : 'bg-cap-blue hover:bg-cap-blue/80 shadow-lg shadow-cap-blue/20',
              )}
            >
              {isStreaming ? <Loader size={16} className="animate-spin" /> : <Send size={16} />}
            </button>
          </div>
          <p className="text-[10px] text-slate-700 text-center mt-1.5">Enter to send · Shift+Enter for newline{isVision && ' · 📎 attach image'}</p>
        </div>
      )}
    </div>
  )
}

// ── Compare mode shared input (handles vision image upload) ──
function CompareInput({ isStreaming, isVision, leftLabel, rightLabel, onSend, input, setInput }: {
  isStreaming: boolean
  isVision: boolean
  leftLabel: string
  rightLabel: string
  onSend: (text: string, imageDataUrl?: string) => void
  input: string
  setInput: (v: string) => void
}) {
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const imageRef = useRef<HTMLInputElement>(null)

  const doSend = () => {
    if ((!input.trim() && !imagePreview) || isStreaming) return
    const img = imagePreview
    onSend(input.trim(), img ?? undefined)
    setImagePreview(null)
  }

  return (
    <div className="p-4 border-t border-white/[0.06] bg-slate-900/30">
      {imagePreview && (
        <div className="mb-2 relative inline-block">
          <img src={imagePreview} alt="upload" className="h-16 w-auto rounded-xl border border-white/10 object-cover" />
          <button onClick={() => setImagePreview(null)}
            className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-slate-900 border border-white/10 text-slate-400 hover:text-red-400 flex items-center justify-center">
            <X size={10} />
          </button>
        </div>
      )}
      <div className={cn(
        'flex gap-2 items-end p-3 rounded-2xl border bg-slate-900/80 transition-colors',
        'border-indigo-500/20 focus-within:border-indigo-500/40',
      )}>
        {isVision && (
          <>
            <input ref={imageRef} type="file" accept="image/*" className="hidden"
              onChange={e => {
                const f = e.target.files?.[0]
                if (f) { const r = new FileReader(); r.onload = ev => setImagePreview(ev.target?.result as string); r.readAsDataURL(f) }
                e.target.value = ''
              }}
            />
            <button onClick={() => imageRef.current?.click()}
              className="p-1.5 rounded-lg text-slate-500 hover:text-purple-400 hover:bg-purple-500/10 transition-colors shrink-0" title="Upload image">
              <ImageIcon size={15} />
            </button>
          </>
        )}
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); doSend() } }}
          placeholder={isVision ? 'Ask both models about the image...' : 'Send to both models simultaneously...'}
          rows={1}
          className="flex-1 bg-transparent text-sm text-slate-200 placeholder-slate-600 resize-none focus:outline-none py-1.5"
          style={{ maxHeight: '120px', overflowY: 'auto' }}
        />
        <button onClick={doSend} disabled={(!input.trim() && !imagePreview) || isStreaming}
          className="p-2.5 bg-indigo-500/20 hover:bg-indigo-500/30 border border-indigo-500/30 rounded-xl text-indigo-400 transition-colors disabled:opacity-50 shrink-0">
          {isStreaming ? <Loader size={16} className="animate-spin" /> : <Send size={16} />}
        </button>
      </div>
      <p className="text-[10px] text-slate-600 mt-1.5 text-center">
        Comparing <span className="text-slate-400">{leftLabel}</span>
        {' '}vs <span className="text-slate-400">{rightLabel}</span>
      </p>
    </div>
  )
}

// ── Main Screen ──
export default function ChatScreen() {
  const {
    loadedModel, compareMode, compareModel,
    isStreaming, params, chatModelType,
    setLoadedModel, setCompareModel, setCompareMode, setIsStreaming,
    setChatModelType,
    activeThreadId, setActiveThreadId,
  } = useChatStore()

  const [settingsOpen, setSettingsOpen] = useState(false)
  const [loadCustomOpen, setLoadCustomOpen] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  // Compare mode uses fully independent local state — never shares thread/activeThread
  // so activeThreadId changes don't wipe compare responses mid-generation
  const [leftCompareMessages,  setLeftCompareMessages]  = useState<Message[]>([])
  const [rightCompareMessages, setRightCompareMessages] = useState<Message[]>([])
  const [compareMessages, setCompareMessages] = useState<Message[]>([])  // kept for compat
  const [compareInput, setCompareInput] = useState('')
  const [isLoadingModel, setIsLoadingModel] = useState(false)
  const [loadingStatus, setLoadingStatus] = useState('')
  const [isLoadingCompareModel, setIsLoadingCompareModel] = useState(false)
  const [compareModelReady, setCompareModelReady] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  // Live thread list from Dexie
  const threads = useLiveQuery(
    () => chatDb.threads.orderBy('updatedAt').reverse().toArray(),
    [], [] as Thread[],
  )

  // Load messages when thread changes
  useEffect(() => {
    if (!activeThreadId) { setMessages([]); return }
    getMessages(activeThreadId).then(setMessages)
  }, [activeThreadId])

  // On mount: sync store with real backend state.
  // If backend was restarted, clear stale loadedModel so user must re-select.
  useEffect(() => {
    if (!loadedModel) return
    apiFetch<{ is_loaded: boolean }>('/api/inference/status')
      .then(s => { if (!s.is_loaded) setLoadedModel(null) })
      .catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleNewThread = async () => {
    const thread = await createThread(loadedModel ?? undefined)
    setActiveThreadId(thread.id)
    setMessages([])
  }

  const handleSelectThread = async (id: string) => {
    setActiveThreadId(id)
    const msgs = await getMessages(id)
    setMessages(msgs)
  }

  const handleDeleteThread = async (id: string) => {
    await deleteThread(id)
    if (activeThreadId === id) {
      setActiveThreadId(null)
      setMessages([])
    }
  }

  const handleSelectModel = useCallback(async (model: string) => {
    // Check real backend status before skipping — don't rely on persisted store state
    // (backend may have been restarted while frontend kept loadedModel in store)
    if (model === loadedModel && !isLoadingModel) {
      try {
        const status = await apiFetch<{ is_loaded: boolean }>('/api/inference/status')
        if (status.is_loaded) {
          toast.info('Model already loaded and ready.')
          return
        }
        // Backend lost the model (restart) — fall through to reload
      } catch { /* fall through */ }
    }
    setLoadedModel(model)
    if (isMockMode()) {
      setIsLoadingModel(true)
      setLoadingStatus('Loading model...')
      setTimeout(() => { setIsLoadingModel(false); setLoadingStatus('') }, 1000)
      return
    }
    setIsLoadingModel(true)
    // Show appropriate loading message based on model type
    const isAdapter = model.includes('outputs') || model.includes('adapter')
    const isMerged = model.includes('exports')
    const isLocal = model.startsWith('C:') || model.startsWith('D:') || model.startsWith('/')
    setLoadingStatus(
      isAdapter ? 'Loading base model + applying LoRA adapter...'
      : isMerged ? 'Loading merged model into VRAM...'
      : isLocal ? 'Loading local model into VRAM...'
      : `Downloading & loading ${modelDisplayName(model)}...`
    )
    try {
      const res = await apiFetch<{
        status: string
        training_system_prompt?: string
        training_inference_params?: Record<string, number>
      }>('/api/inference/load', {
        method: 'POST',
        body: JSON.stringify({ model_path: model, max_seq_length: 2048, load_in_4bit: true }),
      })

      // Pre-populate Settings from training metadata if available
      const trainPrompt  = res.training_system_prompt  || ''
      const trainParams  = res.training_inference_params || {}
      const hasMetadata  = trainPrompt || Object.keys(trainParams).length > 0

      if (hasMetadata) {
        setParams({
          ...(trainPrompt ? { systemPrompt: trainPrompt } : {}),
          ...(trainParams.temperature  !== undefined ? { temperature:       trainParams.temperature        } : {}),
          ...(trainParams.max_tokens   !== undefined ? { maxTokens:         trainParams.max_tokens         } : {}),
          ...(trainParams.repetition_penalty !== undefined ? { repetitionPenalty: trainParams.repetition_penalty } : {}),
          ...(trainParams.top_p        !== undefined ? { topP:              trainParams.top_p              } : {}),
          ...(trainParams.top_k        !== undefined ? { topK:              trainParams.top_k              } : {}),
        })
        toast.info(
          `Settings pre-populated from training config.${trainPrompt ? ' System prompt loaded — edit in ⚙ Settings.' : ''}`,
          { duration: 4000 }
        )
      }
    } catch (err) {
      console.error('Model load error:', err)
    } finally {
      setIsLoadingModel(false)
      setLoadingStatus('')
    }
  }, [])

  // Load right-panel compare model into separate VRAM slot
  const handleSelectCompareModel = useCallback(async (model: string) => {
    // Verify backend status before skipping — backend may have restarted
    if (model === compareModel && compareModelReady) {
      try {
        const status = await apiFetch<{ is_loaded: boolean }>('/api/inference/status-compare')
        if (status.is_loaded) {
          toast.info('Right model already loaded and ready.')
          return
        }
      } catch { /* fall through */ }
      setCompareModelReady(false)
    }
    setCompareModel(model)
    setCompareModelReady(false)
    setRightCompareMessages([])
    if (isMockMode()) {
      setIsLoadingCompareModel(true)
      setTimeout(() => { setIsLoadingCompareModel(false); setCompareModelReady(true) }, 1000)
      return
    }
    setIsLoadingCompareModel(true)
    try {
      const res = await apiFetch<{ status: string; error?: string }>(
        '/api/inference/load-compare',
        { method: 'POST', body: JSON.stringify({ model_path: model, max_seq_length: 2048, load_in_4bit: true }) }
      )
      if (res.status === 'error') {
        toast.error(res.error || 'Failed to load compare model')
      } else {
        setCompareModelReady(true)
        toast.success(`Right model ready: ${model.split(/[\\/]/).pop()}`)
      }
    } catch (err: unknown) {
      toast.error((err as { message?: string })?.message ?? 'Failed to load compare model')
    } finally {
      setIsLoadingCompareModel(false)
    }
  }, [setCompareModel])

  // Auto-create thread if none selected
  const ensureThread = useCallback(async (): Promise<string> => {
    if (activeThreadId) return activeThreadId
    const thread = await createThread(loadedModel ?? undefined)
    setActiveThreadId(thread.id)
    return thread.id
  }, [activeThreadId, loadedModel, setActiveThreadId])

  const sendMessage = useCallback(async (
    text: string,
    targetMessages: Message[],
    setTargetMessages: React.Dispatch<React.SetStateAction<Message[]>>,
    modelName: string | null,
    useAdapter?: boolean | null,
    compareSlot?: boolean,
    imageDataUrl?: string,   // base64 data URL — only sent for the current turn
  ) => {
    const threadId = await ensureThread()

    // Add user message — store text + imageDataUrl so it shows in thread history
    const displayText = text || (imageDataUrl ? '📎 Image' : '')
    const userMsg = await addMessage({
      threadId,
      role: 'user',
      content: displayText,
      modelName: modelName ?? undefined,
      imageDataUrl: imageDataUrl ?? undefined,
    })
    setTargetMessages(prev => [...prev, userMsg])

    if (targetMessages.length === 0) {
      const title = text.slice(0, 40) + (text.length > 40 ? '...' : '')
      await updateThreadTitle(threadId, title)
    }

    // Add streaming assistant message placeholder
    const assistantMsg = await addMessage({ threadId, role: 'assistant', content: '', modelName: modelName ?? undefined, streaming: true })
    setTargetMessages(prev => [...prev, assistantMsg])

    setIsStreaming(true)
    abortRef.current = new AbortController()
    let accumulated = ''

    const finish = async () => {
      await updateMessage(assistantMsg.id, { content: accumulated, streaming: false })
      setTargetMessages(prev => prev.map(m =>
        m.id === assistantMsg.id ? { ...m, content: accumulated, streaming: false } : m,
      ))
      setIsStreaming(false)
    }

    if (isMockMode()) {
      // Demo mode — simulated response
      await streamDemoResponse(
        text,
        (chunk) => {
          accumulated += chunk
          updateMessage(assistantMsg.id, { content: accumulated })
          setTargetMessages(prev => prev.map(m =>
            m.id === assistantMsg.id ? { ...m, content: accumulated } : m,
          ))
        },
        finish,
        abortRef.current.signal,
      )
      return
    }

    // Real mode — SSE stream from backend
    try {
      const history: { role: 'user' | 'assistant'; content: unknown }[] = targetMessages
        .filter(m => m.role === 'user' || m.role === 'assistant')
        .map(m => ({ role: m.role as 'user' | 'assistant', content: m.content }))

      // Current turn: if image attached, use multimodal content (OpenAI vision format)
      const currentContent: unknown = imageDataUrl
        ? [
            { type: 'image_url', image_url: { url: imageDataUrl } },
            { type: 'text', text: text || '' },
          ]
        : text
      history.push({ role: 'user', content: currentContent })

      const body: Record<string, unknown> = {
        messages: history,
        stream: true,
        temperature: params.temperature,  // 0 = greedy, >0 = sampling
        top_p: params.topP,
        top_k: params.topK,
        min_p: params.minP,
        max_tokens: params.maxTokens,
        repetition_penalty: params.repetitionPenalty,
        system: params.systemPrompt || null,
      }
      if (useAdapter !== undefined && useAdapter !== null) {
        body.use_adapter = useAdapter
      }
      if (compareSlot) {
        body.compare_slot = true
      }

      // Read token from localStorage (same as apiFetch does internally)
      const _raw = localStorage.getItem('unslothcraft-auth')
      const _token = _raw ? (JSON.parse(_raw)?.state?.accessToken ?? null) : null
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (_token) headers['Authorization'] = `Bearer ${_token}`

      const resp = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: abortRef.current.signal,
      })

      if (!resp.ok || !resp.body) {
        const errText = await resp.text().catch(() => 'Unknown error')
        throw new Error(errText)
      }

      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6).trim()
          if (data === '[DONE]') break
          try {
            const chunk = JSON.parse(data)
            const content = chunk?.choices?.[0]?.delta?.content ?? ''
            if (content) {
              accumulated += content
              updateMessage(assistantMsg.id, { content: accumulated })
              setTargetMessages(prev => prev.map(m =>
                m.id === assistantMsg.id ? { ...m, content: accumulated } : m,
              ))
            }
          } catch { /* skip malformed chunks */ }
        }
      }
    } catch (err: unknown) {
      if ((err as Error)?.name !== 'AbortError') {
        const msg = (err as Error)?.message ?? 'Inference error'
        accumulated = accumulated || `Error: ${msg}`
        updateMessage(assistantMsg.id, { content: accumulated })
        setTargetMessages(prev => prev.map(m =>
          m.id === assistantMsg.id ? { ...m, content: accumulated } : m,
        ))
      }
    } finally {
      await finish()
    }
  }, [ensureThread, setIsStreaming, params])

  const handleSend = useCallback(async (text: string, imageDataUrl?: string) => {
    await sendMessage(text, messages, setMessages, loadedModel, undefined, false, imageDataUrl)
  }, [sendMessage, messages, loadedModel])

  // Dedicated streaming helper for compare mode — bypasses thread system entirely
  // userMsg is pre-added by handleCompareSend so both panels show it simultaneously
  const streamCompare = useCallback(async (
    text: string,
    setMsgs: React.Dispatch<React.SetStateAction<Message[]>>,
    modelName: string | null,
    compareSlot: boolean,
    asstId: string,
    imageDataUrl?: string,
  ) => {
    // Add streaming assistant placeholder (user msg already added)
    const asstMsg: Message = { id: asstId, role: 'assistant', content: '', createdAt: Date.now() }
    setMsgs(prev => [...prev, asstMsg])

    const currentContent: unknown = imageDataUrl
      ? [{ type: 'image_url', image_url: { url: imageDataUrl } }, { type: 'text', text: text || '' }]
      : text
    const history = [{ role: 'user' as const, content: currentContent }]
    const body: Record<string, unknown> = {
      messages: history, stream: true,
      temperature: params.temperature, top_p: params.topP, top_k: params.topK,
      min_p: params.minP, max_tokens: params.maxTokens,
      repetition_penalty: params.repetitionPenalty,
      system: params.systemPrompt || null,
      compare_slot: compareSlot,
    }
    const _raw = localStorage.getItem('unslothcraft-auth')
    const _token = _raw ? (JSON.parse(_raw)?.state?.accessToken ?? null) : null
    const headers: Record<string, string> = { 'Content-Type': 'application/json' }
    if (_token) headers['Authorization'] = `Bearer ${_token}`

    try {
      const resp = await fetch('/v1/chat/completions', { method: 'POST', headers, body: JSON.stringify(body) })
      if (!resp.ok || !resp.body) {
        const errText = await resp.text().catch(() => 'Request failed')
        setMsgs(prev => prev.map(m => m.id === asstId ? { ...m, content: `⚠ ${errText}` } : m))
        return
      }
      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buffer = '', accumulated = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6).trim()
          if (data === '[DONE]') break
          try {
            const chunk = JSON.parse(data)
            const token = chunk.choices?.[0]?.delta?.content ?? ''
            if (token) {
              accumulated += token
              setMsgs(prev => prev.map(m => m.id === asstMsg.id ? { ...m, content: accumulated } : m))
            }
          } catch { /* skip */ }
        }
      }
    } catch (err) {
      setMsgs(prev => prev.map(m => m.id === asstMsg.id ? { ...m, content: '⚠ Error: ' + String(err) } : m))
    }
  }, [params])

  const handleCompareSend = useCallback(async (text: string, imageDataUrl?: string) => {
    const ts = Date.now()
    const displayText = text || (imageDataUrl ? '📎 Image' : '')
    // Step 1: show user message in BOTH panels immediately — before any generation
    const leftUser:  Message = { id: `u-l-${ts}`, role: 'user', content: displayText, createdAt: ts }
    const rightUser: Message = { id: `u-r-${ts}`, role: 'user', content: displayText, createdAt: ts }
    setLeftCompareMessages(prev  => [...prev,  leftUser])
    setRightCompareMessages(prev => [...prev, rightUser])

    // Step 2: generate responses one by one (both models stay in VRAM)
    await streamCompare(text, setLeftCompareMessages,  loadedModel,  false, `a-l-${ts}`, imageDataUrl)
    await streamCompare(text, setRightCompareMessages, compareModel, true,  `a-r-${ts}`, imageDataUrl)
  }, [sendMessage, messages, compareMessages, loadedModel, compareModel])

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <NavBar />

      <div className="flex-1 flex overflow-hidden min-h-0">
        {/* Sidebar */}
        <ThreadSidebar
          threads={threads}
          activeId={activeThreadId}
          onSelect={handleSelectThread}
          onCreate={handleNewThread}
          onDelete={handleDeleteThread}
        />

        {/* Main area */}
        <div className="flex-1 flex flex-col min-w-0">

          {/* ── Top bar: type + mode + model selectors ── */}
          <div className="flex items-center gap-2 px-4 py-2.5 border-b border-white/[0.06] bg-slate-900/30 flex-wrap">

            {/* Text / Vision toggle */}
            <div className="flex rounded-lg border border-white/[0.08] overflow-hidden text-xs shrink-0">
              {(['text', 'vision'] as const).map(t => (
                <button key={t} onClick={() => setChatModelType(t)}
                  className={cn('flex items-center gap-1.5 px-3 py-1.5 transition-colors capitalize',
                    chatModelType === t ? 'bg-cap-cyan/10 text-cap-cyan' : 'text-slate-500 hover:text-slate-300')}>
                  {t === 'vision' ? <Eye size={11} /> : <Type size={11} />}
                  {t}
                </button>
              ))}
            </div>

            {/* Single / Compare toggle */}
            <div className="flex rounded-lg border border-white/[0.08] overflow-hidden text-xs shrink-0">
              <button onClick={() => setCompareMode(false)}
                className={cn('flex items-center gap-1.5 px-3 py-1.5 transition-colors',
                  !compareMode ? 'bg-slate-700/60 text-slate-200' : 'text-slate-500 hover:text-slate-300')}>
                <MessageSquare size={11} /> Single
              </button>
              <button onClick={() => { setCompareMode(true); setLeftCompareMessages([]); setRightCompareMessages([]) }}
                className={cn('flex items-center gap-1.5 px-3 py-1.5 transition-colors',
                  compareMode ? 'bg-indigo-500/20 text-indigo-300' : 'text-slate-500 hover:text-slate-300')}>
                <Columns2 size={11} /> Compare
              </button>
            </div>

            <div className="w-px h-5 bg-white/[0.08] shrink-0" />

            {/* Model selector(s) */}
            {!compareMode ? (
              <div className="flex items-center gap-2 flex-1 min-w-0">
                <ModelSelector selected={loadedModel} onSelect={handleSelectModel} modelType={chatModelType} label="Select Model" />
                {isLoadingModel && (
                  <div className="flex items-center gap-1.5 text-xs text-slate-400">
                    <Loader size={12} className="animate-spin text-cap-cyan shrink-0" />
                    <span className="truncate text-[11px]">{loadingStatus || 'Loading...'}</span>
                  </div>
                )}
                {loadedModel && !isLoadingModel && (
                  <span className="text-[10px] text-emerald-400 flex items-center gap-1 shrink-0">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block" /> Ready
                  </span>
                )}
              </div>
            ) : (
              <div className="flex items-center gap-2 flex-1 min-w-0 flex-wrap">
                {/* Left model — fine-tuned (LoRA on) */}
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] text-slate-600">Left:</span>
                  <ModelSelector selected={loadedModel} onSelect={handleSelectModel} modelType={chatModelType} label="Left model" />
                </div>
                <span className="text-slate-600 text-xs">vs</span>
                {/* Right — independent model, loaded separately into VRAM */}
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] text-slate-600">Right:</span>
                  <ModelSelector
                    selected={compareModel}
                    onSelect={handleSelectCompareModel}
                    modelType={chatModelType}
                    label="Right model"
                  />
                  {isLoadingCompareModel && (
                    <div className="flex items-center gap-1 text-[11px] text-slate-400">
                      <Loader size={11} className="animate-spin text-cap-cyan" />
                      Loading...
                    </div>
                  )}
                  {compareModel && !isLoadingCompareModel && compareModelReady && (
                    <span className="text-[10px] text-emerald-500">● Ready</span>
                  )}
                </div>
                {isLoadingModel && (
                  <div className="flex items-center gap-1.5 text-xs text-slate-400">
                    <Loader size={12} className="animate-spin text-cap-cyan" />
                    <span className="text-[11px] truncate">{loadingStatus || 'Loading...'}</span>
                  </div>
                )}
              </div>
            )}

            {/* Load custom model */}
            <button onClick={() => setLoadCustomOpen(true)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-800/40 border border-white/[0.06] text-slate-400 hover:text-cap-cyan hover:border-cap-cyan/30 transition-colors text-xs shrink-0"
              title="Load custom model from local path (adapter, GGUF, or merged folder)"
            >
              <Upload size={13} /> Load
            </button>

            {/* Settings */}
            <button onClick={() => setSettingsOpen(true)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-800/40 border border-white/[0.06] text-slate-400 hover:text-slate-200 hover:border-white/15 transition-colors text-xs shrink-0">
              <Settings size={13} /> Settings
            </button>
          </div>

          {/* Chat area */}
          {!compareMode ? (
            <ChatPanel
              messages={messages}
              isStreaming={isStreaming}
              modelName={loadedModel}
              onSend={handleSend}
              disabled={!loadedModel && !isLoadingModel}
              isVision={chatModelType === 'vision'}
            />
          ) : (
            <div className="flex-1 flex flex-col min-h-0">
              {/* Split panels — independent local state, no thread interference */}
              <div className="flex-1 flex overflow-hidden min-h-0 divide-x divide-white/[0.06]">
                <ChatPanel
                  messages={leftCompareMessages}
                  isStreaming={isStreaming}
                  modelName={loadedModel}
                  onSend={() => {}}
                  isCompare
                  isVision={chatModelType === 'vision'}
                />
                <ChatPanel
                  messages={rightCompareMessages}
                  isStreaming={isStreaming}
                  modelName={compareModel}
                  onSend={() => {}}
                  isCompare
                  isVision={chatModelType === 'vision'}
                />
              </div>

              {/* Shared input */}
              <CompareInput
                isStreaming={isStreaming}
                isVision={chatModelType === 'vision'}
                leftLabel={modelDisplayName(loadedModel) || '?'}
                rightLabel={modelDisplayName(compareModel) || '?'}
                onSend={(text) => { handleCompareSend(text); setCompareInput('') }}
                input={compareInput}
                setInput={setCompareInput}
              />
            </div>
          )}
        </div>
      </div>

      <SettingsSheet open={settingsOpen} onClose={() => setSettingsOpen(false)} />

      <AnimatePresence>
        {loadCustomOpen && (
          <LoadCustomModelModal
            open={loadCustomOpen}
            onClose={() => setLoadCustomOpen(false)}
            onLoad={handleSelectModel}
          />
        )}
      </AnimatePresence>
    </div>
  )
}
