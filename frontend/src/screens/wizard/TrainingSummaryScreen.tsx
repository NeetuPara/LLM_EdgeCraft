import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import {
  Brain, Database, Settings, Edit2, Rocket,
  CheckCircle, Cpu, Clock, Tag,
} from 'lucide-react'
import { motion } from 'framer-motion'
import WizardShell from './WizardShell'
import { useTrainingConfigStore } from '@/stores/training-config-store'
import { useTrainingRuntimeStore } from '@/stores/training-runtime-store'
import { trainingApi } from '@/api/training-api'
import { apiFetch } from '@/api/client'
import { isMockMode } from '@/api/mock'
import { cn } from '@/utils/cn'

// ── Summary row ──
function SummaryRow({ label, value }: { label: string; value: string | number | null | undefined }) {
  if (!value && value !== 0) return null
  return (
    <div className="flex items-center justify-between py-1.5 text-sm">
      <span className="text-slate-500">{label}</span>
      <span className="text-slate-200 font-medium text-right">{String(value)}</span>
    </div>
  )
}

// ── Summary card ──
function SummaryCard({
  icon: Icon, title, color, editPath, children,
}: {
  icon: React.ElementType; title: string; color: string
  editPath: string; children: React.ReactNode
}) {
  const navigate = useNavigate()
  return (
    <div className={cn(
      'glass-card p-5 border-l-2',
      `border-l-${color}`,
    )}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2.5">
          <div className={cn('p-2 rounded-lg', `bg-${color}/10`)}>
            <Icon size={15} className={`text-${color}`} />
          </div>
          <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
        </div>
        <button
          onClick={() => navigate(editPath)}
          className="flex items-center gap-1 text-xs text-slate-500 hover:text-cap-cyan transition-colors"
        >
          <Edit2 size={11} />
          Edit
        </button>
      </div>
      <div className="divide-y divide-white/[0.04]">
        {children}
      </div>
    </div>
  )
}

// Shared VRAM + time estimate utilities
function getParamBillions(name: string): number {
  const lower = name.toLowerCase()
  const mB = lower.match(/(\d+(?:\.\d+)?)\s*b(?!\w)/)
  if (mB) return parseFloat(mB[1])
  const mM = lower.match(/(\d+(?:\.\d+)?)\s*m(?!\w)/)
  if (mM) return parseFloat(mM[1]) / 1000
  return 4  // default estimate if unknown
}

function estimateResources(
  config: ReturnType<typeof useTrainingConfigStore.getState>,
  datasetRows?: number,
) {
  const params   = getParamBillions(config.modelName)
  const isVision = config.modelType === 'vision'

  // ── VRAM estimate ────────────────────────────────────────────────────────
  // Text: ~0.55 GB/B params as base weight; VLM adds vision encoder (~0.8 GB)
  // and PyTorch allocator + image preprocessing overhead (0.8 GB/sample + 1.5 GB fixed).
  const baseGb  = Math.max(0.5, params * 0.55)
  const mult    = config.trainingMethod === 'qlora' ? 1.2
                : config.trainingMethod === 'lora'  ? 2.2
                : 4.5
  let vramGb = Math.ceil(baseGb * mult)

  if (isVision) {
    const visionEncoderGb   = 0.8                              // SigLIP/CLIP bf16
    const batchImageOverhead = (config.batchSize || 2) * 0.8   // 0.8 GB/sample
    const fixedVlmOverhead  = 1.5                              // allocator + kernel buffers
    vramGb = Math.ceil(vramGb + visionEncoderGb + batchImageOverhead + fixedVlmOverhead)
  }

  // ── Time estimate ────────────────────────────────────────────────────────
  const epochs       = config.maxSteps > 0 ? 1 : config.numEpochs
  const effectiveBatch = (config.batchSize || 2) * (config.gradAccumSteps || 4)
  const rowsPerEpoch = (datasetRows && datasetRows > 0) ? datasetRows : 1000
  const estSteps     = config.maxSteps > 0
    ? config.maxSteps
    : Math.ceil((rowsPerEpoch / effectiveBatch) * epochs)

  // VLM is ~2× slower than text: vision encoder forward + image token processing.
  // Empirically: SmolVLM-500M on RTX 5080 Laptop → ~16 steps/min.
  // Text model (Gemma-3-1B) on same GPU → ~27 steps/min.
  const baseStepsPerMin = config.trainingMethod === 'full'
    ? 2
    : isVision
      ? (params <= 1 ? 15 : params <= 3 ? 8 : 4)   // VLM: ~2× slower
      : (params <= 1 ? 30 : params <= 3 ? 18 : params <= 7 ? 10 : 5)

  const minutes = Math.ceil(estSteps / baseStepsPerMin)

  return { vramGb, minutes, estSteps }
}

export default function TrainingSummaryScreen() {
  const navigate = useNavigate()
  const config = useTrainingConfigStore()
  const { reset: resetRuntime, hydrateFromStatus } = useTrainingRuntimeStore()
  const [launching, setLaunching] = useState(false)
  const [realVramGb, setRealVramGb] = useState<number | null>(null)

  // Real VRAM estimate from backend
  useEffect(() => {
    if (!config.modelName || isMockMode()) return
    apiFetch<{ total_gb: number }>('/api/models/vram-estimate', {
      method: 'POST',
      body: JSON.stringify({
        model_name: config.modelName,
        training_method: config.trainingMethod,
        lora_rank: config.loraR,
        target_modules: (config.targetModules || '').split(',').map(s => s.trim()).filter(Boolean),
        batch_size: config.batchSize,
        max_seq_length: config.maxSeqLength,
        optimizer: config.optimizer,
        gradient_checkpointing: config.gradientCheckpointing,
        load_in_4bit: config.trainingMethod === 'qlora',
        hf_token: config.hfToken || null,
      }),
    })
      .then(res => setRealVramGb(res.total_gb))
      .catch(() => setRealVramGb(null))
  }, [config.modelName])

  const { vramGb: heuristicVramGb, minutes, estSteps } = estimateResources(config, config.datasetRows || undefined)
  const vramGb = realVramGb ?? heuristicVramGb

  const handleLaunch = async () => {
    setLaunching(true)
    try {
      const isQlora = config.trainingMethod === 'qlora'
      const isLora = config.trainingMethod === 'lora'
      const isFull = config.trainingMethod === 'full'

      const res = await trainingApi.start({
        model_name: config.modelName,
        // Correct field names matching backend's TrainingStartRequest:
        hf_dataset: config.datasetSource === 'huggingface' ? config.datasetName : '',
        local_datasets: config.datasetSource === 'local' ? [config.datasetName] : undefined,
        format_type: config.formatType,
        is_dataset_image: config.modelType === 'vision',
        custom_format_mapping: Object.keys(config.columnMapping ?? {}).length > 0
          ? config.columnMapping
          : undefined,
        system_prompt: config.systemPrompt?.trim() || undefined,
        // Only pass VLM-specific fields when actually training a vision model
        // — prevents stale imageColumn from previous VLM session triggering VLM path
        image_column:    config.modelType === 'vision' ? (config.imageColumn?.trim() || undefined) : undefined,
        dataset_base_dir: config.modelType === 'vision' ? (config.datasetBaseDir?.trim() || undefined) : undefined,
        // Backend expects "LoRA/QLoRA" for any LoRA variant, or anything else for full
        training_type: isFull ? 'full' : 'LoRA/QLoRA',
        load_in_4bit: isQlora,
        use_lora: !isFull,
        max_seq_length: config.maxSeqLength,
        num_epochs: config.numEpochs,
        learning_rate: String(config.learningRate),
        batch_size: config.batchSize,
        gradient_accumulation_steps: config.gradAccumSteps,
        warmup_steps: config.warmupSteps,
        max_steps: config.maxSteps,
        weight_decay: config.weightDecay,
        lora_r: config.loraR,
        lora_alpha: config.loraAlpha,
        lora_dropout: config.loraDropout,
        target_modules: config.targetModules || undefined,
        gradient_checkpointing: config.gradientCheckpointing,
        use_rslora: config.useRslora,
        save_steps: config.saveSteps,
        output_dir: config.outputModelName.trim() || undefined,
        save_strategy: config.saveStrategy,
        early_stopping_patience: config.earlyStoppingPatience,
        train_on_completions: config.trainOnCompletions,
        packing: config.packing,
        optim: config.optimizer,
        lr_scheduler_type: config.lrScheduler,
        enable_wandb: config.enableWandB,
        wandb_project: config.wandbProject,
        enable_tensorboard: config.enableTensorBoard,
        hf_token: config.hfToken || undefined,
        finetune_vision_layers: config.finetuneVisionLayers,
        finetune_language_layers: config.finetuneLanguageLayers,
        finetune_attention_modules: config.finetuneAttentionModules,
        finetune_mlp_modules: config.finetuneMlpModules,
      })

      // Check if the response indicates an error (backend returns 200 with status:"error")
      if (res && (res as { status?: string }).status === 'error') {
        const msg = (res as { message?: string }).message || 'Training failed to start'
        toast.error(msg)
        setLaunching(false)
        return
      }

      resetRuntime()
      hydrateFromStatus({
        phase: 'starting',
        current_step: 0, total_steps: 0,
        current_epoch: 0, total_epochs: config.numEpochs,
        progress_percent: 0, eta_seconds: 0,
        status_message: 'Initializing training...',
      })

      toast.success('Training started!')
      navigate('/training')
    } catch (err: unknown) {
      const msg = (err as { message?: string })?.message || 'Failed to start training'
      console.error('Training start failed:', err)
      toast.error(msg)
      setLaunching(false)
    }
  }

  const canLaunch = !!config.modelName && !!config.datasetName

  return (
    <WizardShell
      step={4}
      title="Ready to Launch"
      description="Review your configuration and start training."
      onBack={() => navigate('/new/params')}
      onNext={handleLaunch}
      nextLabel="Start Training"
      nextLoading={launching}
      nextDisabled={!canLaunch}
    >
      <div className="space-y-4">

        {/* ── Fine-tuned Model Name ── */}
        <motion.div initial={{ opacity: 1, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0 }}>
          <div className="glass-card p-5 border-l-2 border-l-cap-cyan">
            <div className="flex items-center gap-2.5 mb-3">
              <div className="p-2 rounded-lg bg-cap-cyan/10">
                <Tag size={15} className="text-cap-cyan" />
              </div>
              <h3 className="text-sm font-semibold text-slate-200">Fine-tuned Model Name</h3>
              <span className="ml-auto text-[10px] text-slate-500">Used as folder name in outputs/</span>
            </div>
            <input
              value={config.outputModelName}
              onChange={e => config.patch({ outputModelName: e.target.value })}
              placeholder={`e.g. ${config.modelName ? config.modelName.split('/').pop() + '_finetuned' : 'my_model_finetuned'}`}
              className="glass-input text-sm py-2.5 font-mono w-full"
            />
            <p className="text-[10px] mt-1.5">
              {config.outputModelName.trim()
                ? <span className="text-slate-400">Saved to: <span className="font-mono text-cap-cyan">~/.unslothcraft/outputs/{config.outputModelName.trim().replace(/[^\w\-.]/g, '_')}/</span></span>
                : <span className="text-slate-600">Leave blank to auto-name from model + timestamp</span>
              }
            </p>
          </div>
        </motion.div>

        {/* ── Config cards ── */}
        <motion.div initial={{ opacity: 1, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
          <SummaryCard icon={Brain} title="Model" color="cap-cyan" editPath="/new/model">
            <SummaryRow label="Base Model" value={config.modelName || '(not set)'} />
            <SummaryRow label="Type" value={config.modelType} />
            <SummaryRow label="Method" value={config.trainingMethod.toUpperCase()} />
            <SummaryRow label="Quantization" value={config.trainingMethod === 'qlora' ? '4-bit NF4' : config.trainingMethod === 'lora' ? '16-bit' : 'Full precision'} />
          </SummaryCard>
        </motion.div>

        <motion.div initial={{ opacity: 1, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <SummaryCard icon={Database} title="Dataset" color="emerald-400" editPath="/new/dataset">
            <SummaryRow label="Dataset" value={config.datasetName ? config.datasetName.replace(/\\/g, '/').split('/').pop()! : '(not set)'} />
            <SummaryRow label="Source" value={config.datasetSource} />
            <SummaryRow label="Split" value={config.datasetSplit} />
            <SummaryRow label="Format" value={config.formatType} />
            {config.modelType === 'vision' && config.imageColumn && (
              <SummaryRow label="Image Column" value={config.imageColumn} />
            )}
            {config.modelType === 'vision' && (
              <SummaryRow label="Vision Encoder" value={config.finetuneVisionLayers ? 'Fine-tuned' : 'Frozen'} />
            )}
          </SummaryCard>
        </motion.div>

        <motion.div initial={{ opacity: 1, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
          <SummaryCard icon={Settings} title="Hyperparameters" color="indigo-400" editPath="/new/params">
            <SummaryRow label={config.maxSteps > 0 ? 'Max Steps' : 'Epochs'} value={config.maxSteps > 0 ? config.maxSteps : config.numEpochs} />
            <SummaryRow label="Learning Rate" value={Number(config.learningRate).toExponential(1)} />
            <SummaryRow label="Batch Size" value={config.batchSize} />
            <SummaryRow label="Context Length" value={`${config.maxSeqLength} tokens`} />
            <SummaryRow label="LoRA r / α" value={`${config.loraR} / ${config.loraAlpha}`} />
            <SummaryRow label="Optimizer" value={config.optimizer} />
          </SummaryCard>
        </motion.div>

        {/* ── Resource estimates ── */}
        <motion.div initial={{ opacity: 1, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
          <div className="glass-card p-5">
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-cap-cyan/10 border border-cap-cyan/20">
                  <Cpu size={16} className="text-cap-cyan" />
                </div>
                <div>
                  <p className="text-xs text-slate-500">Est. VRAM</p>
                  <p className="text-xl font-bold text-slate-100 font-display">{vramGb} GB</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20">
                  <Clock size={16} className="text-amber-400" />
                </div>
                <div>
                  <p className="text-xs text-slate-500">Est. Duration</p>
                  <p className="text-xl font-bold text-slate-100 font-display">
                    {minutes >= 60 ? `${(minutes / 60).toFixed(1)}h` : `${minutes}m`}
                  </p>
                </div>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-white/[0.06]">
              {[
                config.trainingMethod === 'qlora' && '4-bit quantization active — lowest VRAM usage',
                config.packing && 'Sample packing enabled — better GPU utilization',
                config.trainOnCompletions && 'Training on completions only',
                config.enableWandB && `WandB logging → project: ${config.wandbProject}`,
              ].filter(Boolean).map((note, i) => (
                <div key={i} className="flex items-center gap-2 text-xs text-slate-500 py-0.5">
                  <CheckCircle size={11} className="text-emerald-500 shrink-0" />
                  {note}
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* ── Missing config warning ── */}
        {(!config.modelName || !config.datasetName) && (
          <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-4 text-sm text-amber-300 space-y-1">
            {!config.modelName && <p>⚠ No model selected — go back to Step 1</p>}
            {!config.datasetName && <p>⚠ No dataset selected — go back to Step 2</p>}
          </div>
        )}

        {/* ── Launch hint ── */}
        <div className="flex items-center gap-2 text-xs text-slate-600 justify-center pt-2">
          <Rocket size={12} />
          Training will start in a background subprocess — you can navigate away safely
        </div>

      </div>
    </WizardShell>
  )
}
