import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { ChevronDown, ChevronRight, Cpu, BarChart2, Activity, Upload, Download, FileCode } from 'lucide-react'
import { toast } from 'sonner'
import WizardShell from './WizardShell'
import InfoTooltip from '@/components/InfoTooltip'
import { useTrainingConfigStore } from '@/stores/training-config-store'
import { apiFetch } from '@/api/client'
import { isMockMode } from '@/api/mock'
import { cn } from '@/utils/cn'
import { LR_SCHEDULERS, OPTIMIZERS } from '@/config/constants'

// ── Reusable number field ──
function NumberField({
  label, value, onChange, min, max, step = 1, tooltip, suffix,
}: {
  label: string; value: number; onChange: (v: number) => void
  min?: number; max?: number; step?: number; tooltip?: string; suffix?: string
}) {
  return (
    <div>
      <label className="flex items-center gap-1.5 text-xs text-slate-500 mb-1.5">
        {label}
        {tooltip && <InfoTooltip text={tooltip} size={12} />}
      </label>
      <div className="relative">
        <input
          type="number"
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={e => onChange(Number(e.target.value))}
          className="glass-input text-sm py-2.5 pr-10"
        />
        {suffix && (
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-slate-500">{suffix}</span>
        )}
      </div>
    </div>
  )
}

// ── Select field ──
function SelectField({
  label, value, onChange, options, tooltip,
}: {
  label: string; value: string; onChange: (v: string) => void
  options: readonly { id: string; label: string }[]; tooltip?: string
}) {
  return (
    <div>
      <label className="flex items-center gap-1.5 text-xs text-slate-500 mb-1.5">
        {label}
        {tooltip && <InfoTooltip text={tooltip} size={12} />}
      </label>
      <div className="relative">
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          className="glass-input text-sm py-2.5 appearance-none pr-8"
        >
          {options.map(o => <option key={o.id} value={o.id}>{o.label}</option>)}
        </select>
        <ChevronDown size={13} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
      </div>
    </div>
  )
}

// ── Toggle ──
function Toggle({
  label, checked, onChange, tooltip,
}: {
  label: string; checked: boolean; onChange: (v: boolean) => void; tooltip?: string
}) {
  return (
    <div className="flex items-center justify-between py-2">
      <span className="flex items-center gap-1.5 text-sm text-slate-300">
        {label}
        {tooltip && <InfoTooltip text={tooltip} size={12} />}
      </span>
      <button
        onClick={() => onChange(!checked)}
        className={cn(
          'relative w-10 h-5 rounded-full transition-colors duration-200',
          checked ? 'bg-cap-cyan' : 'bg-slate-700',
        )}
      >
        <div className={cn(
          'absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform duration-200',
          checked && 'translate-x-5',
        )} />
      </button>
    </div>
  )
}

// ── Section card ──
function Section({ title, icon: Icon, children, defaultOpen = true }: {
  title: string; icon: React.ElementType; children: React.ReactNode; defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="glass-card p-0 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2.5 px-5 py-4 text-sm font-semibold text-slate-300 hover:bg-white/[0.02] transition-colors"
      >
        <Icon size={15} className="text-slate-500" />
        {title}
        <ChevronRight size={14} className={cn('ml-auto text-slate-600 transition-transform', open && 'rotate-90')} />
      </button>
      {open && (
        <div className="px-5 pb-5 border-t border-white/[0.06]">
          <div className="pt-4">
            {children}
          </div>
        </div>
      )}
    </div>
  )
}

interface VramBreakdown {
  total_gb: number
  breakdown: {
    model_weights_gb: number
    lora_adapters_gb: number
    optimizer_states_gb: number
    gradients_gb: number
    activations_gb: number
    cuda_overhead_gb: number
  }
}

function buildYamlTemplate(config: ReturnType<typeof useTrainingConfigStore.getState>): string {
  return `# ── EdgeCraft Training Configuration ──
# Upload this file in the Hyperparameters page to auto-fill all fields.
# Only keys you provide will be applied. All other values stay unchanged.

# ── Essential ──
num_epochs: ${config.numEpochs}          # Full passes over the dataset (1 = safe default)
max_steps: ${config.maxSteps}            # Override epochs: stop after N steps (0 = use epochs)
learning_rate: ${config.learningRate}    # 2e-4 for QLoRA · 5e-5 for LoRA
batch_size: ${config.batchSize}          # Per-GPU samples per step
max_seq_length: ${config.maxSeqLength}   # Context window in tokens (2048 · 4096 · 8192)
lr_scheduler_type: ${config.lrScheduler} # cosine | linear | constant | polynomial

# ── LoRA ──
lora_r: ${config.loraR}                  # Rank (4–256). Higher = more capacity, more VRAM.
lora_alpha: ${config.loraAlpha}          # Scaling = alpha/r. Usually equal to rank.
lora_dropout: ${config.loraDropout}      # Dropout on LoRA layers (0 recommended)
use_rslora: ${config.useRslora}          # Rank-Stabilised LoRA — recommended at r≥32
train_on_completions: ${config.trainOnCompletions}  # Only learn assistant responses

# ── Advanced ──
gradient_accumulation_steps: ${config.gradAccumSteps}  # Effective batch = batch_size × grad_accum
warmup_steps: ${config.warmupSteps}      # LR warmup for first N steps
weight_decay: ${config.weightDecay}      # L2 regularisation (0.01 = standard)
save_steps: ${config.saveSteps}          # Checkpoint every N steps (0 = end only)
optimizer: ${config.optimizer}           # adamw_8bit | adamw_torch | sgd | lion_8bit
packing: ${config.packing}              # Pack multiple short samples (30–50% speedup for Q&A)
gradient_checkpointing: ${config.gradientCheckpointing}  # unsloth | true | none
`
}

export default function HyperparamsScreen() {
  const navigate = useNavigate()
  const config = useTrainingConfigStore()
  const { patch, setHighestStep, modelName, trainingMethod, modelType } = config
  const isVlm = modelType === 'vision'
  const configFileRef = useRef<HTMLInputElement>(null)
  const [importing, setImporting] = useState(false)
  const [lastImport, setLastImport] = useState<{ filename: string; count: number } | null>(null)

  useEffect(() => { setHighestStep(2) }, [setHighestStep])

  const [useMaxSteps, setUseMaxSteps] = useState(config.maxSteps > 0)
  const [vramData, setVramData] = useState<VramBreakdown | null>(null)
  const [vramLoading, setVramLoading] = useState(false)

  // Fetch real VRAM estimate whenever relevant params change (debounced 600ms)
  useEffect(() => {
    if (!modelName || isMockMode()) return
    const t = setTimeout(async () => {
      setVramLoading(true)
      try {
        const res = await apiFetch<VramBreakdown>('/api/models/vram-estimate', {
          method: 'POST',
          body: JSON.stringify({
            model_name: modelName,
            training_method: trainingMethod,
            lora_rank: config.loraR,
            target_modules: (config.targetModules || '').split(',').map((s: string) => s.trim()).filter(Boolean),
            batch_size: config.batchSize,
            max_seq_length: config.maxSeqLength,
            optimizer: config.optimizer,
            gradient_checkpointing: config.gradientCheckpointing,
            load_in_4bit: trainingMethod === 'qlora',
            hf_token: config.hfToken || null,
          }),
        })
        setVramData(res)
      } catch {
        setVramData(null)
      } finally {
        setVramLoading(false)
      }
    }, 600)
    return () => clearTimeout(t)
  }, [modelName, trainingMethod, config.loraR, config.batchSize, config.maxSeqLength,
      config.optimizer, config.gradientCheckpointing, config.targetModules])

  const vramLabel = vramLoading
    ? '…'
    : vramData
      ? `${vramData.total_gb} GB`
      : '—'

  const handleConfigImport = async (file: File) => {
    if (isMockMode()) {
      toast.info('Config import is not available in demo mode')
      return
    }
    setImporting(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await apiFetch<{ config: Record<string, unknown>; applied_count: number; filename: string }>(
        '/api/config/parse', { method: 'POST', body: form }
      )
      // Apply all returned values to the store
      // Coerce numeric fields that might come back as strings from YAML parsing
      const coerced = { ...res.config }
      const numericFields = ['learningRate', 'batchSize', 'numEpochs', 'maxSteps',
        'maxSeqLength', 'gradAccumSteps', 'warmupSteps', 'weightDecay',
        'loraR', 'loraAlpha', 'loraDropout', 'saveSteps', 'evalSteps', 'earlyStoppingPatience']
      for (const f of numericFields) {
        if (f in coerced && coerced[f] !== undefined && coerced[f] !== null) {
          const n = Number(coerced[f])
          if (!isNaN(n)) coerced[f] = n
        }
      }
      patch(coerced as Parameters<typeof patch>[0])
      // Sync useMaxSteps toggle with parsed value
      if ('maxSteps' in coerced) setUseMaxSteps((coerced.maxSteps as number) > 0)
      setLastImport({ filename: res.filename, count: res.applied_count })
      toast.success(`Config loaded — ${res.applied_count} values applied from "${res.filename}"`)
    } catch (err: unknown) {
      toast.error((err as { message?: string })?.message ?? 'Failed to parse config file')
    } finally {
      setImporting(false)
    }
  }

  const handleDownloadTemplate = () => {
    const yaml = buildYamlTemplate(useTrainingConfigStore.getState())
    const blob = new Blob([yaml], { type: 'text/yaml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'training_config.yaml'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <WizardShell
      step={3}
      title="Hyperparameters"
      description="Tune your training settings. Defaults work well for most use cases."
      onBack={() => navigate('/new/dataset')}
      onNext={() => { setHighestStep(3); navigate('/new/summary') }}
      footer={
        <div className="flex items-center gap-1.5 text-xs text-slate-500 group relative">
          <Cpu size={12} />
          Est. VRAM:
          <span className={cn('font-semibold', vramLoading ? 'text-slate-500 animate-pulse' : 'text-cap-cyan')}>
            {vramLabel}
          </span>
          {vramData && (
            <div className="absolute bottom-6 left-0 hidden group-hover:flex flex-col gap-1 bg-slate-900 border border-white/10 rounded-xl p-3 text-[10px] text-slate-400 w-52 shadow-xl z-10">
              <p className="text-slate-200 font-semibold text-xs mb-1">VRAM Breakdown</p>
              {Object.entries(vramData.breakdown).filter(([k]) => k !== 'total_gb').map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span>{k.replace(/_gb$/, '').replace(/_/g, ' ')}</span>
                  <span className="text-slate-200">{(v as number).toFixed(2)} GB</span>
                </div>
              ))}
              <div className="flex justify-between border-t border-white/10 pt-1 mt-0.5 text-cap-cyan font-semibold">
                <span>Total</span>
                <span>{vramData.total_gb} GB</span>
              </div>
            </div>
          )}
        </div>
      }
    >
      <div className="space-y-4">

        {/* ── Config file import ── */}
        <div className="glass-card p-4">
          <input
            ref={configFileRef}
            type="file"
            accept=".yaml,.yml,.cfg,.ini,.txt"
            className="hidden"
            onChange={e => {
              const file = e.target.files?.[0]
              if (file) handleConfigImport(file)
              e.target.value = ''
            }}
          />
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-cap-cyan/10">
              <FileCode size={15} className="text-cap-cyan" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-slate-300">Import Config File</p>
              <p className="text-xs text-slate-500 mt-0.5">
                {lastImport
                  ? <span className="text-emerald-400">✓ {lastImport.count} values loaded from "{lastImport.filename}"</span>
                  : 'Upload .yaml · .cfg · .txt to auto-fill all fields below'
                }
              </p>
            </div>
            <button
              onClick={handleDownloadTemplate}
              className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-cap-cyan transition-colors px-3 py-1.5 rounded-lg border border-white/[0.08] hover:border-cap-cyan/30"
              title="Download YAML template with current values"
            >
              <Download size={12} />
              Template
            </button>
            <button
              onClick={() => configFileRef.current?.click()}
              disabled={importing}
              className="flex items-center gap-1.5 text-xs btn-primary px-3 py-1.5 rounded-lg disabled:opacity-50"
            >
              <Upload size={12} />
              {importing ? 'Parsing…' : 'Import'}
            </button>
          </div>
        </div>

        {/* ── Essential ── */}
        <Section title="Essential" icon={Activity} defaultOpen>
          <div className="grid grid-cols-2 gap-4">

            {/* Epochs / Max Steps toggle */}
            <div className="col-span-2">
              <div className="flex items-center gap-3 mb-3">
                <span className="text-xs text-slate-500">Training limit:</span>
                <div className="flex rounded-lg border border-white/[0.08] overflow-hidden text-xs">
                  {[false, true].map(isSteps => (
                    <button
                      key={String(isSteps)}
                      onClick={() => {
                        setUseMaxSteps(isSteps)
                        if (!isSteps) patch({ maxSteps: 0 })
                        else patch({ numEpochs: 0 })
                      }}
                      className={cn(
                        'px-3 py-1.5 transition-colors',
                        useMaxSteps === isSteps
                          ? 'bg-cap-cyan/10 text-cap-cyan'
                          : 'text-slate-500 hover:text-slate-300',
                      )}
                    >
                      {isSteps ? 'Max Steps' : 'Epochs'}
                    </button>
                  ))}
                </div>
              </div>
              {!useMaxSteps ? (
                <NumberField
                  label="Number of Epochs"
                  value={config.numEpochs}
                  onChange={v => patch({ numEpochs: v })}
                  min={1} max={100} step={1}
                  tooltip="Full passes over the dataset. 1 epoch for large datasets (50K+ rows). 2–3 for medium (5K–50K). 3–5 for small (<5K). More epochs = more learning but risk overfitting."
                />
              ) : (
                <NumberField
                  label="Max Steps"
                  value={config.maxSteps}
                  onChange={v => patch({ maxSteps: v })}
                  min={1}
                  tooltip="Stop after N optimizer steps regardless of epochs. Use 30–100 for quick tests, 500–2000 for real training."
                />
              )}
            </div>

            <NumberField
              label="Learning Rate"
              value={config.learningRate}
              onChange={v => patch({ learningRate: v })}
              step={1e-5}
              min={1e-6} max={1e-2}
              tooltip="2e-4 is optimal for QLoRA (4-bit). Use 5e-5 to 1e-4 for LoRA (16-bit) since gradients are more precise. Too high → unstable, too low → slow learning."
            />
            <NumberField
              label="Batch Size"
              value={config.batchSize}
              onChange={v => patch({ batchSize: v })}
              min={1} max={64}
              tooltip="Samples per step. With grad_accum=4, effective batch = 2×4 = 8. Increase for more stable gradients (uses more VRAM). Reduce to 1 if you hit OOM."
            />
            <NumberField
              label="LoRA Rank (r)"
              value={config.loraR}
              onChange={v => {
                patch({ loraR: v, loraAlpha: v })  // Keep alpha = rank by default
              }}
              min={4} max={256} step={4}
              tooltip="Capacity of the LoRA adapters. r=16: fast experiments. r=32: recommended for <7B. r=64: maximum quality. Alpha auto-syncs to match rank."
            />
            <NumberField
              label="LoRA Alpha"
              value={config.loraAlpha}
              onChange={v => patch({ loraAlpha: v })}
              min={1} max={512}
              tooltip="Scaling = alpha/r. Set equal to rank (1:1) for standard adaptation. Set to 2×rank for stronger LoRA effect. Currently: alpha/r = {(config.loraAlpha / Math.max(config.loraR, 1)).toFixed(1)}."
            />
            <NumberField
              label="Context Length"
              value={config.maxSeqLength}
              onChange={v => patch({ maxSeqLength: v })}
              min={256} max={131072} step={256}
              tooltip="2048 for short Q&A (alpaca-style). 4096 for conversations & code. 8192+ for long documents. VRAM scales quadratically — 4096 uses ~4× more than 2048."
              suffix="tokens"
            />
            <SelectField
              label="LR Scheduler"
              value={config.lrScheduler}
              onChange={v => patch({ lrScheduler: v })}
              options={LR_SCHEDULERS}
              tooltip="Cosine: best for most tasks — gradual decay to near-zero. Linear: simpler, good for short runs. Constant: use with max_steps for quick experiments."
            />
            {/* RSLoRA + Train on Completions — recommended, shown in Essential */}
            <div className="col-span-2 pt-3 border-t border-white/[0.06] space-y-1">
              <Toggle
                label="RSLoRA (Rank-Stabilized)"
                checked={config.useRslora}
                onChange={v => patch({ useRslora: v })}
                tooltip="Uses √r scaling instead of r for gradient normalization. Recommended ON when r≥32 (your r={config.loraR}). Free quality improvement — no speed or VRAM cost."
              />
              <Toggle
                label="Train on Completions Only"
                checked={config.trainOnCompletions}
                onChange={v => patch({ trainOnCompletions: v })}
                tooltip="Compute loss only on assistant responses, not on instruction tokens. Model focuses 100% on learning HOW to answer. Recommended ON for all instruction/chat datasets."
              />
            </div>
          </div>
        </Section>

        {/* ── Advanced ── */}
        <Section title="Advanced" icon={BarChart2} defaultOpen={false}>
          <div className="grid grid-cols-2 gap-4">

            {/* Model output name */}
            <div className="col-span-2">
              <label className="flex items-center gap-1.5 text-xs text-slate-500 mb-1.5">
                Output Model Name
                <InfoTooltip text="Name for your fine-tuned model. Used as the folder name in outputs/ and shown in Chat Sandbox to select for inference. Leave blank to auto-generate from model name + timestamp." size={12} />
              </label>
              <input
                value={config.outputModelName}
                onChange={e => patch({ outputModelName: e.target.value })}
                placeholder="e.g. gemma3_1B_TOS"
                className={`glass-input text-sm py-2.5 font-mono ${!config.outputModelName.trim() ? 'border-amber-500/40' : ''}`}
              />
              {config.outputModelName.trim()
                ? <p className="text-[10px] text-slate-500 mt-1">Saved to: <span className="font-mono text-slate-400">~/.unslothcraft/outputs/{config.outputModelName.trim().replace(/[^\w\-.]/g, '_')}/</span></p>
                : <p className="text-[10px] text-slate-600 mt-1">Optional — leave blank to auto-name from model + timestamp</p>
              }
            </div>

            <NumberField
              label="Gradient Accumulation"
              value={config.gradAccumSteps}
              onChange={v => patch({ gradAccumSteps: v })}
              min={1} max={64}
              tooltip="Accumulate gradients over N steps before updating. Simulates larger batch size."
            />
            <NumberField
              label="Warmup Steps"
              value={config.warmupSteps}
              onChange={v => patch({ warmupSteps: v })}
              min={0} max={1000}
              tooltip="Linear LR warmup for the first N steps. Helps stability at the start."
            />
            <NumberField
              label="Weight Decay"
              value={config.weightDecay}
              onChange={v => patch({ weightDecay: v })}
              min={0} max={0.5} step={0.01}
              tooltip="L2 regularization strength. Small values (0.01) help prevent overfitting."
            />
            <NumberField
              label="LoRA Dropout"
              value={config.loraDropout}
              onChange={v => patch({ loraDropout: v })}
              min={0} max={0.5} step={0.05}
              tooltip="Dropout on LoRA layers. Usually 0 — Unsloth's kernels don't support non-zero well."
            />
            <div className="col-span-2">
              <label className="flex items-center gap-1.5 text-xs text-slate-500 mb-2">
                Checkpoint Saving
                <InfoTooltip text="'Best epoch' requires eval data — saves the epoch with lowest eval loss (early stopping). 'Every epoch' saves after each epoch. 'Last only' saves once at the end." size={12} />
              </label>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { value: 'no',    label: 'Last Only',    desc: 'Save once at end' },
                  { value: 'epoch', label: 'Every Epoch',  desc: 'Save after each epoch' },
                  { value: 'best',  label: 'Best Epoch',   desc: 'Lowest eval loss only' },
                ] .map(opt => (
                  <button
                    key={opt.value}
                    onClick={() => patch({ saveStrategy: opt.value as 'no' | 'epoch' | 'steps' })}
                    className={cn(
                      'flex flex-col items-center gap-0.5 px-3 py-2.5 rounded-xl border text-xs transition-all',
                      config.saveStrategy === opt.value
                        ? 'bg-cap-cyan/10 border-cap-cyan/30 text-cap-cyan'
                        : 'bg-slate-800/30 border-white/[0.06] text-slate-400 hover:border-white/15',
                    )}
                  >
                    <span className="font-medium">{opt.label}</span>
                    <span className="text-[10px] opacity-60">{opt.desc}</span>
                  </button>
                ))}
              </div>
              {config.saveStrategy === 'best' && (
                <NumberField
                  label="Early Stopping Patience"
                  value={config.earlyStoppingPatience}
                  onChange={v => patch({ earlyStoppingPatience: v })}
                  min={1} max={10} step={1}
                  tooltip="Stop training if eval loss doesn't improve for this many consecutive epochs. e.g. patience=3 → stops after 3 epochs of no improvement."
                />
              )}
            </div>
            <SelectField
              label="Optimizer"
              value={config.optimizer}
              onChange={v => patch({ optimizer: v })}
              options={OPTIMIZERS}
              tooltip="AdamW 8-bit is recommended — same quality as FP32 AdamW with 75% less VRAM."
            />

            <div className="col-span-2 pt-2 border-t border-white/[0.06]">
              <Toggle
                label="Sample Packing"
                checked={config.packing}
                onChange={v => patch({ packing: v })}
                tooltip="Packs multiple short samples into one sequence to eliminate padding waste. Speeds up training 30–50% on short Q&A datasets. Avoid for long documents or code."
              />
            </div>
          </div>
        </Section>

        {/* Vision Layers — shown only for VLM models */}
        {isVlm && (
          <Section title="Vision Layers" icon={Activity} defaultOpen>
            <div className="col-span-2 space-y-1 text-xs text-slate-500 mb-3 leading-relaxed">
              Control which parts of the VLM are fine-tuned. Freezing the vision encoder
              saves VRAM and trains faster — recommended unless your images differ significantly
              from ImageNet (e.g. medical scans, satellite, thermal).
            </div>
            <div className="col-span-2 space-y-1">
              <Toggle
                label="Fine-tune Vision Encoder"
                checked={config.finetuneVisionLayers}
                onChange={v => patch({ finetuneVisionLayers: v })}
                tooltip="Train the image feature extractor. OFF = frozen (faster, lower VRAM). ON = adapts to domain-specific visuals (e.g. solar panels, X-rays)."
              />
              <Toggle
                label="Fine-tune Language Layers"
                checked={config.finetuneLanguageLayers}
                onChange={v => patch({ finetuneLanguageLayers: v })}
                tooltip="Train the language decoder. Keep ON for most VLM fine-tuning."
              />
              <Toggle
                label="Fine-tune Attention Modules"
                checked={config.finetuneAttentionModules}
                onChange={v => patch({ finetuneAttentionModules: v })}
                tooltip="LoRA on Q/K/V/O projections. Recommended ON."
              />
              <Toggle
                label="Fine-tune MLP Modules"
                checked={config.finetuneMlpModules}
                onChange={v => patch({ finetuneMlpModules: v })}
                tooltip="LoRA on feed-forward layers. Recommended ON for domain adaptation."
              />
            </div>
          </Section>
        )}

        {/* Experiment Tracking (WandB + TensorBoard) — hidden until implemented
        <Section title="Experiment Tracking" icon={Activity} defaultOpen={false}>
          ...WandB and TensorBoard toggles...
        </Section>
        */}

      </div>
    </WizardShell>
  )
}
