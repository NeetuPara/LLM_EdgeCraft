import { useState, useEffect } from 'react'
import {
  Link, Layers, Package, ChevronDown, Folder,
  CheckCircle, Loader, Download, CloudUpload,
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import NavBar from '@/components/NavBar'
import InfoTooltip from '@/components/InfoTooltip'
import { MOCK_RUNS, MOCK_CHECKPOINTS, MOCK_LORAS } from '@/api/mock/data'
import { GGUF_QUANT_OPTIONS } from '@/config/constants'
import { isMockMode } from '@/api/mock'
import { apiFetch } from '@/api/client'
import { cn } from '@/utils/cn'

type ExportMethod = 'lora' | 'merged_16bit' | 'gguf'

const METHODS: {
  id: ExportMethod; label: string; desc: string; size: string; icon: React.ElementType; color: string
}[] = [
  {
    id: 'lora',       label: 'LoRA Adapter',   icon: Link,   color: 'cap-cyan',
    desc: 'Save adapter weights only. Requires the base model to run.',
    size: '~50–200 MB',
  },
  {
    id: 'merged_16bit', label: 'Merged Model', icon: Layers, color: 'indigo-400',
    desc: 'Merge LoRA into the base model (bfloat16). Standalone, ready to deploy.',
    size: 'Same as base model',
  },
  {
    id: 'gguf',       label: 'GGUF',           icon: Package, color: 'amber-400',
    desc: 'Quantized format for Ollama, LM Studio, llama.cpp.',
    size: 'Variable by quant level',
  },
]

type ExportStatus = 'idle' | 'loading' | 'exporting' | 'done' | 'error'

interface CheckpointItem {
  path: string
  display_name: string
  run_name: string
  loss?: number | null
  base_model?: string | null
  peft_type?: string | null
  is_final?: boolean
}

export default function ExportScreen() {
  // ── Checkpoint list ──
  const [checkpoints, setCheckpoints] = useState<CheckpointItem[]>([])
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>('')
  const [loadingCheckpoints, setLoadingCheckpoints] = useState(false)

  // ── Export config ──
  const [method, setMethod] = useState<ExportMethod>('gguf')
  const [quantLevels, setQuantLevels] = useState<string[]>(['q4_k_m'])
  const [destType, setDestType] = useState<'local' | 'hub'>('local')
  const [localPath, setLocalPath] = useState('')
  const [hubRepo, setHubRepo] = useState('')
  const [hubPrivate, setHubPrivate] = useState(true)
  const [hfToken, setHfToken] = useState('')

  // ── Status ──
  const [status, setStatus] = useState<ExportStatus>('idle')
  const [progress, setProgress] = useState(0)
  const [logLines, setLogLines] = useState<string[]>([])

  const addLog = (msg: string) => setLogLines(prev => [...prev, msg])

  // ── Load real checkpoints on mount ──
  useEffect(() => {
    if (isMockMode()) {
      // Demo: combine mock runs + loras as checkpoint list
      const items: CheckpointItem[] = [
        ...MOCK_CHECKPOINTS.map(cp => ({
          path: cp.path,
          display_name: cp.is_final ? 'Final model' : `Step ${cp.step}`,
          run_name: cp.run_id,
          loss: cp.loss,
          is_final: cp.is_final,
        })),
        ...MOCK_LORAS.map(l => ({
          path: l.path,
          display_name: l.name,
          run_name: l.name,
          base_model: l.base_model,
        })),
      ]
      setCheckpoints(items)
      setSelectedCheckpoint(items[0]?.path ?? '')
      return
    }

    setLoadingCheckpoints(true)
    apiFetch<CheckpointItem[]>('/api/export/checkpoints')
      .then(data => {
        setCheckpoints(data)
        setSelectedCheckpoint(data[0]?.path ?? '')
      })
      .catch(() => setCheckpoints([]))
      .finally(() => setLoadingCheckpoints(false))
  }, [])

  const toggleQuant = (id: string) => {
    setQuantLevels(prev =>
      prev.includes(id) ? prev.filter(q => q !== id) : [...prev, id],
    )
  }

  const selectedItem = checkpoints.find(c => c.path === selectedCheckpoint)

  // ── Export ──
  const handleExport = async () => {
    if (status === 'exporting' || status === 'loading') return
    setStatus('loading')
    setLogLines([])
    setProgress(0)

    if (isMockMode()) {
      await runDemoExport()
      return
    }

    await runRealExport()
  }

  const runRealExport = async () => {
    try {
      // Step 1: load checkpoint
      addLog(`Loading checkpoint: ${selectedCheckpoint}...`)
      setProgress(5)

      const loaded = await apiFetch<{ success: boolean; message: string; is_peft: boolean }>(
        '/api/export/load-checkpoint',
        {
          method: 'POST',
          body: JSON.stringify({
            checkpoint_path: selectedCheckpoint,
            max_seq_length: 2048,
            load_in_4bit: true,
          }),
        },
      )

      if (!loaded.success) throw new Error(loaded.message ?? 'Failed to load checkpoint')
      addLog(`Checkpoint loaded${loaded.is_peft ? ' (LoRA adapter detected)' : ''}`)
      setProgress(15)
      setStatus('exporting')

      const saveDir = localPath.trim() || ''
      const pushToHub = destType === 'hub'

      // Step 2: run export(s)
      if (method === 'lora') {
        addLog('Saving LoRA adapter weights...')
        const res = await apiFetch<{ success: boolean; message: string; save_directory: string }>(
          '/api/export/export/lora',
          {
            method: 'POST',
            body: JSON.stringify({
              save_directory: saveDir,
              push_to_hub: pushToHub,
              repo_id: hubRepo || null,
              hf_token: hfToken || null,
              private: hubPrivate,
            }),
          },
        )
        setProgress(90)
        addLog(res.message)
        addLog(`Saved to: ${res.save_directory}`)

      } else if (method === 'merged_16bit') {
        addLog('Merging LoRA into base model (fp16)...')
        const res = await apiFetch<{ success: boolean; message: string; save_directory: string }>(
          '/api/export/export/merged',
          {
            method: 'POST',
            body: JSON.stringify({
              save_directory: saveDir,
              format_type: '16-bit (FP16)',
              push_to_hub: pushToHub,
              repo_id: hubRepo || null,
              hf_token: hfToken || null,
              private: hubPrivate,
            }),
          },
        )
        setProgress(90)
        addLog(res.message)
        addLog(`Saved to: ${res.save_directory}`)

      } else if (method === 'gguf') {
        // Export each quant level sequentially
        const totalQuants = quantLevels.length
        for (let i = 0; i < totalQuants; i++) {
          const quant = quantLevels[i]
          addLog(`Exporting GGUF (${quant.toUpperCase()})... [${i + 1}/${totalQuants}]`)
          const res = await apiFetch<{ success: boolean; message: string; save_directory: string }>(
            '/api/export/export/gguf',
            {
              method: 'POST',
              body: JSON.stringify({
                save_directory: saveDir,
                quantization_method: quant.toUpperCase(),
                push_to_hub: pushToHub,
                repo_id: hubRepo || null,
                hf_token: hfToken || null,
              }),
            },
          )
          const pct = 15 + Math.round(75 * (i + 1) / totalQuants)
          setProgress(pct)
          addLog(res.message)
          addLog(`Saved: ${res.save_directory}`)
        }
      }

      setProgress(100)
      addLog('✓ Export complete!')
      setStatus('done')

    } catch (err: unknown) {
      const msg = (err as Error)?.message ?? 'Export failed'
      addLog(`Error: ${msg}`)
      setStatus('error')
      setProgress(0)
    }
  }

  const runDemoExport = async () => {
    await new Promise(r => setTimeout(r, 600))
    addLog('Loading checkpoint from ' + (selectedCheckpoint || './outputs/run_001') + '...')
    setStatus('exporting')
    const steps =
      method === 'gguf' ? [
        { msg: 'Base model loaded in 4-bit NF4', pct: 20 },
        { msg: 'LoRA adapter weights merged (fp16)', pct: 45 },
        { msg: `Running llama.cpp quantization → ${quantLevels[0]?.toUpperCase() ?? 'Q4_K_M'}`, pct: 70 },
        { msg: `Quantization complete: ${quantLevels.length} file(s) created`, pct: 90 },
      ] : method === 'merged_16bit' ? [
        { msg: 'Dequantizing base model to fp16...', pct: 35 },
        { msg: 'Merging LoRA adapters...', pct: 65 },
        { msg: 'Saving merged model (safetensors)...', pct: 88 },
      ] : [
        { msg: 'Saving LoRA adapter weights...', pct: 60 },
        { msg: 'Writing adapter_config.json...', pct: 85 },
      ]

    for (const step of steps) {
      await new Promise(r => setTimeout(r, 600 + Math.random() * 400))
      addLog(step.msg)
      setProgress(step.pct)
    }
    addLog(destType === 'hub'
      ? `Pushed to HuggingFace Hub: ${hubRepo || 'username/model-name'}`
      : `Saved to ${localPath || './exports/'}`,
    )
    await new Promise(r => setTimeout(r, 300))
    setProgress(100)
    addLog('✓ Export complete!')
    setStatus('done')
  }

  const canExport = selectedCheckpoint && (method !== 'gguf' || quantLevels.length > 0)

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <NavBar />
      <div className="flex-1 overflow-y-auto min-h-0">
        <div className="max-w-3xl mx-auto px-6 py-8 space-y-5">

          <motion.div initial={{ opacity: 1, y: 10 }} animate={{ opacity: 1, y: 0 }}>
            <h1 className="text-2xl font-bold text-slate-100 font-display mb-1">Export Model</h1>
            <p className="text-slate-400 text-sm">Export your fine-tuned model in your preferred format.</p>
          </motion.div>

          {/* ── Source selector ── */}
          <div className="glass-card p-5 space-y-4">
            <h2 className="text-sm font-semibold text-slate-300">Source Checkpoint</h2>

            {loadingCheckpoints ? (
              <div className="flex items-center gap-2 text-sm text-slate-500">
                <Loader size={14} className="animate-spin" /> Loading checkpoints...
              </div>
            ) : checkpoints.length === 0 ? (
              <p className="text-sm text-slate-500">
                {isMockMode()
                  ? 'No checkpoints found.'
                  : 'No training runs found. Complete a training run first.'}
              </p>
            ) : (
              <div className="relative">
                <select
                  value={selectedCheckpoint}
                  onChange={e => setSelectedCheckpoint(e.target.value)}
                  className="glass-input text-sm py-2.5 appearance-none pr-8"
                >
                  {checkpoints.map(cp => (
                    <option key={cp.path} value={cp.path}>
                      {cp.run_name} — {cp.display_name}
                      {cp.loss != null ? ` (loss ${Number(cp.loss).toFixed(4)})` : ''}
                    </option>
                  ))}
                </select>
                <ChevronDown size={13} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
              </div>
            )}

            {selectedItem && (
              <div className="flex items-center gap-3 text-xs text-slate-500 flex-wrap">
                {selectedItem.base_model && (
                  <span>Base: <span className="text-slate-300">{selectedItem.base_model.split('/').pop()}</span></span>
                )}
                {selectedItem.loss != null && (
                  <span>Loss: <span className="text-slate-300">{Number(selectedItem.loss).toFixed(4)}</span></span>
                )}
                {selectedItem.peft_type && (
                  <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    {selectedItem.peft_type}
                  </span>
                )}
              </div>
            )}
          </div>

          {/* ── Export method ── */}
          <div className="glass-card p-5">
            <h2 className="text-sm font-semibold text-slate-300 mb-4">Export Method</h2>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {METHODS.map(({ id, label, desc, size, icon: Icon, color }) => (
                <button
                  key={id}
                  onClick={() => setMethod(id)}
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
                    <p className={cn('font-semibold text-sm', method === id ? `text-${color}` : 'text-slate-200')}>
                      {label}
                    </p>
                    <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{desc}</p>
                    <p className="text-xs text-slate-600 mt-1">{size}</p>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* ── GGUF quantization levels ── */}
          <AnimatePresence>
            {method === 'gguf' && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="overflow-hidden"
              >
                <div className="glass-card p-5">
                  <div className="flex items-center gap-2 mb-4">
                    <h2 className="text-sm font-semibold text-slate-300">Quantization Levels</h2>
                    <InfoTooltip text="Select one or more. Multiple levels export sequentially." />
                    <span className="text-xs text-slate-500 ml-auto">{quantLevels.length} selected</span>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                    {GGUF_QUANT_OPTIONS.map(({ id, label, description, recommended }) => (
                      <button
                        key={id}
                        onClick={() => toggleQuant(id)}
                        className={cn(
                          'flex items-center gap-2.5 p-3 rounded-xl border text-left text-sm transition-all',
                          quantLevels.includes(id)
                            ? 'bg-amber-500/10 border-amber-500/30'
                            : 'bg-slate-800/40 border-white/[0.08] hover:border-white/15',
                        )}
                      >
                        <div className={cn(
                          'w-4 h-4 rounded border-2 flex items-center justify-center shrink-0',
                          quantLevels.includes(id) ? 'bg-amber-500 border-amber-500' : 'border-slate-600',
                        )}>
                          {quantLevels.includes(id) && <CheckCircle size={10} className="text-white" />}
                        </div>
                        <div className="min-w-0">
                          <p className={cn('font-mono font-medium', quantLevels.includes(id) ? 'text-amber-400' : 'text-slate-200')}>
                            {label}
                            {recommended && <span className="ml-1.5 text-[9px] text-emerald-400 font-sans">★</span>}
                          </p>
                          <p className="text-[10px] text-slate-600 truncate">{description}</p>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* ── Destination ── */}
          <div className="glass-card p-5 space-y-4">
            <h2 className="text-sm font-semibold text-slate-300">Destination</h2>
            <div className="flex gap-2">
              {([['local', 'Local Directory', Folder], ['hub', 'HuggingFace Hub', CloudUpload]] as const).map(
                ([id, label, Icon]) => (
                  <button
                    key={id}
                    onClick={() => setDestType(id)}
                    className={cn(
                      'flex items-center gap-2 px-4 py-2.5 rounded-xl border text-sm font-medium transition-all',
                      destType === id
                        ? 'bg-cap-cyan/10 border-cap-cyan/30 text-cap-cyan'
                        : 'bg-slate-800/40 border-white/[0.08] text-slate-400 hover:text-slate-200',
                    )}
                  >
                    <Icon size={15} /> {label}
                  </button>
                ),
              )}
            </div>

            {destType === 'local' ? (
              <input
                value={localPath}
                onChange={e => setLocalPath(e.target.value)}
                placeholder="Leave empty for default (~/.unslothcraft/exports/)"
                className="glass-input text-sm font-mono py-2.5"
              />
            ) : (
              <div className="space-y-3">
                <input
                  value={hubRepo}
                  onChange={e => setHubRepo(e.target.value)}
                  placeholder="username/my-fine-tuned-model"
                  className="glass-input text-sm font-mono py-2.5"
                />
                <input
                  type="password"
                  value={hfToken}
                  onChange={e => setHfToken(e.target.value)}
                  placeholder="hf_... (HuggingFace write token)"
                  className="glass-input text-sm font-mono py-2.5"
                />
                <label className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" checked={hubPrivate} onChange={e => setHubPrivate(e.target.checked)}
                    className="accent-cap-cyan" />
                  <span className="text-sm text-slate-400">Private repository</span>
                </label>
              </div>
            )}
          </div>

          {/* ── Export button ── */}
          <motion.button
            onClick={handleExport}
            disabled={!canExport || status === 'exporting' || status === 'loading'}
            className={cn(
              'w-full py-4 rounded-2xl font-bold text-base flex items-center justify-center gap-3 transition-all',
              status === 'done'
                ? 'bg-emerald-500/20 border border-emerald-500/30 text-emerald-400'
                : status === 'error'
                ? 'bg-red-500/20 border border-red-500/30 text-red-400'
                : 'btn-primary',
            )}
            whileHover={{ scale: canExport && status === 'idle' ? 1.01 : 1 }}
            whileTap={{ scale: 0.99 }}
          >
            {status === 'loading' || status === 'exporting' ? (
              <><Loader size={20} className="animate-spin" /> Exporting...</>
            ) : status === 'done' ? (
              <><CheckCircle size={20} /> Export Complete!</>
            ) : status === 'error' ? (
              <><Download size={20} /> Retry Export</>
            ) : (
              <><Download size={20} /> Export Model</>
            )}
          </motion.button>

          {/* ── Progress + log ── */}
          <AnimatePresence>
            {(status === 'exporting' || status === 'loading' || status === 'done' || status === 'error') && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="glass-card p-5 space-y-4"
              >
                {status !== 'error' && (
                  <>
                    <div className="flex items-center justify-between text-xs text-slate-400">
                      <span>Progress</span>
                      <span className={cn('font-semibold', status === 'done' ? 'text-emerald-400' : 'text-cap-cyan')}>
                        {progress}%
                      </span>
                    </div>
                    <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                      <motion.div
                        className={cn('h-full rounded-full', status === 'done' ? 'bg-emerald-400' : 'bg-cap-cyan')}
                        initial={{ width: 0 }}
                        animate={{ width: `${progress}%` }}
                        transition={{ duration: 0.5, ease: 'easeOut' }}
                      />
                    </div>
                  </>
                )}

                <div className="bg-slate-950/60 rounded-xl p-4 font-mono text-xs space-y-1 max-h-40 overflow-y-auto">
                  {logLines.map((line, i) => (
                    <p key={i} className={cn(
                      line.startsWith('✓') ? 'text-emerald-400' :
                      line.startsWith('Error:') ? 'text-red-400' :
                      'text-slate-400',
                    )}>
                      {line.startsWith('✓') || line.startsWith('Error:')
                        ? line
                        : <><span className="text-slate-700 mr-2">›</span>{line}</>}
                    </p>
                  ))}
                  {(status === 'exporting' || status === 'loading') && (
                    <span className="inline-block w-1 h-3.5 bg-cap-cyan animate-pulse" />
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

        </div>
      </div>
    </div>
  )
}
