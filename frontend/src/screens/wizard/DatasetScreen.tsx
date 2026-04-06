import { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Search, Upload, Database, CheckCircle,
  Table, ChevronDown, X, FileText, Loader, Trash2,
  AlertCircle, Sparkles, FolderOpen,
} from 'lucide-react'
import { toast } from 'sonner'
import WizardShell from './WizardShell'
import InfoTooltip from '@/components/InfoTooltip'
import { useTrainingConfigStore } from '@/stores/training-config-store'
import { mockDatasets } from '@/api/mock'
import { isMockMode } from '@/api/mock'
import { apiFetch } from '@/api/client'
import { cn } from '@/utils/cn'

const SPLITS = ['train', 'test', 'validation', 'train[:10%]', 'train[:1000]']
const FORMATS = ['auto', 'alpaca', 'sharegpt', 'chatml', 'custom'] as const

/**
 * Two simple roles the user assigns to each column:
 *   Input  — what the model receives (clause text, image, question…).
 *            Multiple columns → concatenated into the user turn.
 *   Output — what the model should produce (label, explanation, answer…).
 *            Multiple columns → concatenated into the assistant turn.
 *
 * The system prompt is collected separately in the System Prompt textarea.
 * Internally the worker converts these to user/assistant roles before training.
 */
const COLUMN_ROLES     = ['input', 'output'] as const
const VLM_COLUMN_ROLES = ['input', 'output'] as const  // image column is auto-pinned, not in dropdown
type ColumnRole = typeof COLUMN_ROLES[number]

const ROLE_LABELS: Record<string, string> = {
  input:  'Input  (user turn)',
  output: 'Output (assistant)',
  image:  'Image column',
}

type DatasetSource = 'huggingface' | 'local'

interface DatasetResult {
  id: string; name: string; rows: number; format: string; desc: string
}

interface FormatCheckResult {
  format: string
  needs_mapping: boolean
  columns: string[]
  preview_rows: Record<string, unknown>[]
  suggested_mapping?: Record<string, string>
  is_image?: boolean
  image_path_only?: boolean
  detected_image_column?: string
  warning?: string
}

function normaliseResult(raw: Record<string, unknown>): FormatCheckResult {
  return {
    format:               (raw.detected_format ?? raw.format ?? 'unknown') as string,
    needs_mapping:        (raw.requires_manual_mapping ?? raw.needs_mapping ?? false) as boolean,
    columns:              (raw.columns ?? []) as string[],
    preview_rows:         (raw.preview_samples ?? raw.preview_rows ?? []) as Record<string, unknown>[],
    suggested_mapping:    (raw.suggested_mapping ?? undefined) as Record<string, string> | undefined,
    is_image:             (raw.is_image ?? false) as boolean,
    image_path_only:      (raw.image_path_only ?? false) as boolean,
    detected_image_column:(raw.detected_image_column ?? undefined) as string | undefined,
    warning:              (raw.warning ?? undefined) as string | undefined,
  }
}

function isMappingComplete(mapping: Record<string, string>, isVlm = false): boolean {
  const roles = new Set(Object.values(mapping))
  if (isVlm) return roles.has('image') && roles.has('input') && roles.has('output')
  return roles.has('input') && roles.has('output')
}

// ── Column mapping card ────────────────────────────────────────────────────

/** Status card — green/amber summary + current mapping badges. */
function MappingCard({
  mapping,
  autoDetected,
  isVlm = false,
}: {
  mapping: Record<string, string>
  autoDetected: boolean
  isVlm?: boolean
}) {
  const ok = isMappingComplete(mapping, isVlm)
  const imageCols  = Object.entries(mapping).filter(([, r]) => r === 'image').map(([c]) => c)
  const inputCols  = Object.entries(mapping).filter(([, r]) => r === 'input').map(([c]) => c)
  const outputCols = Object.entries(mapping).filter(([, r]) => r === 'output').map(([c]) => c)

  return (
    <div className={cn(
      'rounded-xl border p-4 mb-3',
      ok ? 'bg-emerald-500/10 border-emerald-500/20' : 'bg-amber-500/10 border-amber-500/20',
    )}>
      <div className="flex items-start gap-3">
        <div className={cn('p-1.5 rounded-lg shrink-0 mt-0.5', ok ? 'bg-emerald-500/15' : 'bg-amber-500/15')}>
          {ok
            ? <CheckCircle size={14} className="text-emerald-400" />
            : <AlertCircle size={14} className="text-amber-400" />
          }
        </div>
        <div className="flex-1 min-w-0">
          <p className={cn('text-sm font-semibold', ok ? 'text-emerald-300' : 'text-amber-300')}>
            {ok
              ? (autoDetected ? 'Mapping auto-detected — review the column headers' : 'Column mapping ready')
              : 'Assign Input and Output columns'}
          </p>
          <p className={cn('text-xs mt-0.5 leading-relaxed', ok ? 'text-emerald-400/70' : 'text-amber-400/70')}>
            {ok
              ? autoDetected
                ? 'Heuristics suggested the mapping below. Use the column header dropdowns to adjust.'
                : 'Training will use these columns. Adjust via the column header dropdowns if needed.'
              : 'Use the dropdowns in the column headers below. Assign at least one Input and one Output column.'
            }
          </p>

          {/* Mapping summary */}
          {ok && (
            <div className="mt-2.5 space-y-1">
              {imageCols.length > 0 && (
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className="text-[10px] text-slate-500 w-14 shrink-0">Image:</span>
                  {imageCols.map(c => (
                    <span key={c} className="inline-flex items-center px-2 py-0.5 rounded-md bg-amber-500/10 border border-amber-500/20 text-[11px] font-mono text-amber-300">{c}</span>
                  ))}
                  <span className="text-[10px] text-slate-600">→ vision encoder</span>
                </div>
              )}
              {inputCols.length > 0 && (
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className="text-[10px] text-slate-500 w-14 shrink-0">Input:</span>
                  {inputCols.map(c => (
                    <span key={c} className="inline-flex items-center px-2 py-0.5 rounded-md bg-blue-500/10 border border-blue-500/20 text-[11px] font-mono text-blue-300">{c}</span>
                  ))}
                  <span className="text-[10px] text-slate-600">→ user turn</span>
                </div>
              )}
              {outputCols.length > 0 && (
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className="text-[10px] text-slate-500 w-14 shrink-0">Output:</span>
                  {outputCols.map(c => (
                    <span key={c} className="inline-flex items-center px-2 py-0.5 rounded-md bg-violet-500/10 border border-violet-500/20 text-[11px] font-mono text-violet-300">{c}</span>
                  ))}
                  <span className="text-[10px] text-slate-600">→ assistant turn{outputCols.length > 1 ? ' (concatenated)' : ''}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

/** Role dropdown rendered inside each table column header.
 *  When the column is the pinned image column, shows a locked badge instead. */
function HeaderRolePicker({
  col,
  currentRole,
  onRoleChange,
  locked = false,
}: {
  col: string
  currentRole: string | undefined
  onRoleChange: (col: string, role: string | undefined) => void
  locked?: boolean
}) {
  const COLOR = {
    input:  'border-blue-400/50 text-blue-300 bg-blue-500/10',
    output: 'border-violet-400/50 text-violet-300 bg-violet-500/10',
    image:  'border-amber-400/50 text-amber-300 bg-amber-500/10',
  } as Record<string, string>

  // Locked image column — show non-interactive badge
  if (locked && currentRole === 'image') {
    return (
      <div className="mt-1.5">
        <span className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-md border border-amber-400/40 text-amber-300 bg-amber-500/10">
          🖼 Image (auto-detected)
        </span>
      </div>
    )
  }

  return (
    <div className="relative mt-1.5" onClick={e => e.stopPropagation()}>
      <select
        value={currentRole ?? '_none'}
        onChange={e => onRoleChange(col, e.target.value === '_none' ? undefined : e.target.value)}
        className={cn(
          'appearance-none text-[10px] pl-2 pr-5 py-0.5 rounded-md border cursor-pointer transition-colors',
          'focus:outline-none focus:ring-1 focus:ring-cap-cyan/40',
          currentRole
            ? COLOR[currentRole] ?? 'border-cap-cyan/40 text-cap-cyan bg-cap-cyan/10'
            : 'border-white/10 text-slate-500 bg-slate-900/50 border-dashed',
        )}
      >
        <option value="_none">Assign role…</option>
        {COLUMN_ROLES.map(role => (
          <option key={role} value={role}>{ROLE_LABELS[role]}</option>
        ))}
      </select>
      <ChevronDown size={9} className="absolute right-1.5 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
    </div>
  )
}

// ── Main screen ────────────────────────────────────────────────────────────

export default function DatasetScreen() {
  const navigate = useNavigate()
  const {
    datasetName, datasetSource, datasetSplit, formatType,
    columnMapping, systemPrompt, modelType, imageColumn,
    patch, setHighestStep,
  } = useTrainingConfigStore()

  const isVlmMode = modelType === 'vision'

  const [tab, setTab] = useState<DatasetSource>(datasetSource)
  const [search, setSearch] = useState(datasetName || '')
  const [results, setResults] = useState<DatasetResult[]>([])
  const [searching, setSearching] = useState(false)
  const [selected, setSelected] = useState(datasetName)
  const [checking, setChecking] = useState(false)
  const [formatResult, setFormatResult] = useState<FormatCheckResult | null>(null)
  const [uploading, setUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [localDatasets, setLocalDatasets] = useState<{
    name: string; path: string; rows?: number
    isFolder?: boolean; baseDir?: string; type?: string
  }[]>([])
  const [pendingFile, setPendingFile] = useState<File | null>(null)
  const [conflictName, setConflictName] = useState<string | null>(null)

  // Local mapping state — synced to store on every change
  const [localMapping, setLocalMapping] = useState<Record<string, string>>(columnMapping ?? {})
  const autoDetectedRef = useRef(false)

  useEffect(() => { setHighestStep(1) }, [setHighestStep])

  // Debounced HF search
  useEffect(() => {
    if (tab !== 'huggingface') return
    const t = setTimeout(async () => {
      setSearching(true)
      const res = await mockDatasets.search(search)
      setResults(res as DatasetResult[])
      setSearching(false)
    }, 300)
    return () => clearTimeout(t)
  }, [search, tab])

  const checkFormat = useCallback(async (name: string) => {
    if (!name) return
    setChecking(true)
    setFormatResult(null)
    setLocalMapping({})
    autoDetectedRef.current = false
    try {
      if (isMockMode()) {
        const res = await mockDatasets.checkFormat()
        setFormatResult(normaliseResult(res as Record<string, unknown>))
      } else {
        const res = await apiFetch<Record<string, unknown>>('/api/datasets/check-format', {
          method: 'POST',
          body: JSON.stringify({ dataset_name: name, train_split: 'train', is_vlm: isVlmMode }),
        })
        const norm = normaliseResult(res)
        setFormatResult(norm)

        // Build initial mapping
        let initMapping: Record<string, string> = {}
        const detectedImgCol = norm.detected_image_column

        // For datasets with an image column: auto-pin it as 'image' role (locked badge)
        if (norm.is_image && detectedImgCol) {
          initMapping[detectedImgCol] = 'image'
          patch({ imageColumn: detectedImgCol })
        }

        // Pre-fill text column roles from heuristic suggestion
        // Exclude the image column by name — it's already pinned above
        if (norm.suggested_mapping && Object.keys(norm.suggested_mapping).length > 0) {
          for (const [col, role] of Object.entries(norm.suggested_mapping)) {
            if (col !== detectedImgCol) initMapping[col] = role
          }
          autoDetectedRef.current = true
        }

        if (Object.keys(initMapping).length > 0) {
          setLocalMapping(initMapping)
          patch({ columnMapping: initMapping })
        }

        const rows = (res.total_rows ?? res.rows ?? res.num_rows ?? 0) as number
        if (rows > 0) patch({ datasetRows: rows })
      }
    } catch (err) {
      console.error('Format check failed:', err)
    } finally {
      setChecking(false)
    }
  }, [patch])

  const selectDataset = (name: string) => {
    setSelected(name)
    patch({ datasetName: name, datasetSource: tab })
    checkFormat(name)
  }

  // Fetch local datasets
  useEffect(() => {
    if (isMockMode()) {
      setLocalDatasets([
        { name: 'alpaca_unsloth.json', path: 'alpaca_unsloth.json', rows: 52002 },
        { name: 'my_custom_data.jsonl', path: 'my_custom_data.jsonl', rows: 1500 },
      ])
      return
    }
    apiFetch<{ datasets: { id: string; label: string; path: string; rows?: number; base_dir?: string; is_folder?: boolean; type?: string }[] }>('/api/datasets/local')
      .then(data => setLocalDatasets(data.datasets.map(d => ({
        name: d.label, path: d.path, rows: d.rows,
        isFolder: d.is_folder, baseDir: d.base_dir, type: d.type,
      }))))
      .catch(() => {})
  }, [tab])

  const refreshLocalList = () =>
    apiFetch<{ datasets: { id: string; label: string; path: string; rows?: number; base_dir?: string; is_folder?: boolean; type?: string }[] }>('/api/datasets/local')
      .then(data => setLocalDatasets(data.datasets.map(d => ({
        name: d.label, path: d.path, rows: d.rows,
        isFolder: d.is_folder, baseDir: d.base_dir, type: d.type,
      }))))
      .catch(() => {})

  const handleFileUpload = async (file: File, action: 'ask' | 'replace' | 'new' = 'ask') => {
    if (isMockMode()) {
      selectDataset(file.name)
      toast.success(`Dataset "${file.name}" selected`)
      return
    }
    const allowed = ['.json', '.jsonl', '.csv', '.parquet', '.zip']
    const ext = '.' + file.name.split('.').pop()?.toLowerCase()
    if (!allowed.includes(ext)) {
      toast.error(`Unsupported format. Use: ${allowed.join(', ')}`)
      return
    }
    setUploading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await apiFetch<{
        conflict?: boolean; filename: string; stored_path: string
        size_bytes: number; is_folder?: boolean; base_dir?: string; file_count?: number
      }>(`/api/datasets/upload?action=${action}`, { method: 'POST', body: form })

      if (res.conflict) {
        setPendingFile(file)
        setConflictName(res.filename)
        return
      }

      // Store base_dir for VLM datasets (zip folders)
      if (res.is_folder && res.base_dir) {
        patch({ datasetBaseDir: res.base_dir })
      } else {
        patch({ datasetBaseDir: '' })
      }

      selectDataset(res.stored_path)
      const sizeLabel = res.size_bytes > 1024 * 1024
        ? `${(res.size_bytes / 1024 / 1024).toFixed(1)} MB`
        : `${(res.size_bytes / 1024).toFixed(1)} KB`
      const detail = res.is_folder
        ? `${res.file_count} files, ${sizeLabel}`
        : sizeLabel
      toast.success(`Uploaded: ${res.filename} (${detail})`)
      refreshLocalList()
    } catch (err: unknown) {
      toast.error((err as { message?: string })?.message ?? 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const handleConflictReplace = () => {
    if (!pendingFile) return
    setPendingFile(null); setConflictName(null)
    handleFileUpload(pendingFile, 'replace')
  }

  const handleConflictSaveNew = () => {
    if (!pendingFile) return
    setPendingFile(null); setConflictName(null)
    handleFileUpload(pendingFile, 'new')
  }

  const handleDelete = async (name: string) => {
    try {
      await apiFetch(`/api/datasets/upload/${encodeURIComponent(name)}`, { method: 'DELETE' })
      toast.success(`Deleted: ${name}`)
      if (selected === name) { setSelected(''); patch({ datasetName: '' }) }
      refreshLocalList()
    } catch {
      toast.error('Failed to delete dataset')
    }
  }

  const handleRoleChange = useCallback((col: string, role: string | undefined) => {
    setLocalMapping(prev => {
      const next = { ...prev }
      // Multiple columns can share input/output; image is unique — remove from any other col
      if (role === 'image') {
        for (const [c, r] of Object.entries(next)) { if (r === 'image') delete next[c] }
      }
      delete next[col]
      if (role) next[col] = role
      patch({
        columnMapping: next,
        ...(role === 'image' ? { imageColumn: col } : {}),
        ...(role === undefined && prev[col] === 'image' ? { imageColumn: '' } : {}),
      })
      return next
    })
  }, [patch])

  // Show mapping UI when:
  //   - user explicitly selected "custom" format, OR
  //   - VLM mode (image column must be assigned), OR
  //   - backend says needs_mapping / detected "custom" / returned a heuristic suggestion
  const showMapping = formatType === 'custom' || isVlmMode || (!!formatResult && (
    formatResult.needs_mapping ||
    formatResult.format === 'custom' ||
    !!formatResult.suggested_mapping ||
    !!formatResult.is_image
  ))

  const canProceed = !!selected

  return (
    <WizardShell
      step={2}
      title="Configure Dataset"
      description="Choose the dataset you want to use for fine-tuning."
      onBack={() => navigate('/new/model')}
      onNext={() => { setHighestStep(2); navigate('/new/params') }}
      nextDisabled={!canProceed}
    >
      <div className="space-y-5">

        {/* ── Source tabs ── */}
        <div className="flex gap-2">
          {(['huggingface', 'local'] as DatasetSource[]).map(t => (
            <button
              key={t}
              onClick={() => { setTab(t); patch({ datasetSource: t }) }}
              className={cn(
                'flex items-center gap-2 px-4 py-2.5 rounded-xl border text-sm font-medium transition-all',
                tab === t
                  ? 'bg-cap-cyan/10 border-cap-cyan/30 text-cap-cyan'
                  : 'bg-slate-800/40 border-white/[0.08] text-slate-400 hover:text-slate-200',
              )}
            >
              {t === 'huggingface' ? <Database size={15} /> : <Upload size={15} />}
              {t === 'huggingface' ? 'HuggingFace' : 'Local Upload'}
            </button>
          ))}
        </div>

        {/* ── HuggingFace Search ── */}
        {tab === 'huggingface' && (
          <div className="glass-card p-5 space-y-4">
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-semibold text-slate-300">Dataset</h2>
              <InfoTooltip text="Search any public HuggingFace dataset or type the full ID (e.g. owner/dataset-name)." />
            </div>

            <div className="relative">
              <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
              <input
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search datasets or enter ID..."
                className="glass-input pl-9 pr-9 text-sm"
              />
              {search && (
                <button onClick={() => setSearch('')} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300">
                  <X size={13} />
                </button>
              )}
            </div>

            <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
              {searching && (
                <div className="flex items-center justify-center py-4">
                  <Loader size={16} className="text-cap-cyan animate-spin" />
                </div>
              )}
              {!searching && results.map(r => (
                <button
                  key={r.id}
                  onClick={() => selectDataset(r.name)}
                  className={cn(
                    'w-full flex items-center gap-3 px-4 py-3 rounded-xl border text-sm text-left transition-all',
                    selected === r.name
                      ? 'bg-cap-cyan/10 border-cap-cyan/30'
                      : 'bg-slate-800/30 border-white/[0.06] hover:border-white/15 hover:bg-slate-800/50',
                  )}
                >
                  {selected === r.name
                    ? <CheckCircle size={14} className="text-cap-cyan shrink-0" />
                    : <div className="w-3 h-3 rounded-full border-2 border-slate-600 shrink-0" />
                  }
                  <div className="flex-1 min-w-0">
                    <p className={cn('font-medium truncate', selected === r.name ? 'text-cap-cyan' : 'text-slate-200')}>
                      {r.name}
                    </p>
                    <p className="text-xs text-slate-500 truncate">{r.desc}</p>
                  </div>
                  <div className="text-right shrink-0">
                    <span className="text-xs text-slate-500">{r.rows.toLocaleString()} rows</span>
                    <br />
                    <span className="badge-neutral text-[10px] py-0">{r.format}</span>
                  </div>
                </button>
              ))}
            </div>

            <div className="pt-2 border-t border-white/[0.06]">
              <p className="text-xs text-slate-500 mb-2">Or enter a dataset ID directly:</p>
              <input
                value={selected || ''}
                onChange={e => { setSelected(e.target.value); patch({ datasetName: e.target.value }) }}
                placeholder="owner/dataset-name"
                className="glass-input text-sm font-mono py-2.5"
              />
            </div>
          </div>
        )}

        {/* ── Local Upload ── */}
        {tab === 'local' && (
          <div className="glass-card p-5 space-y-4">
            <input
              ref={fileInputRef}
              type="file"
              accept=".json,.jsonl,.csv,.parquet,.zip"
              className="hidden"
              onChange={e => {
                const file = e.target.files?.[0]
                if (file) handleFileUpload(file)
                e.target.value = ''
              }}
            />

            <div
              className="border-2 border-dashed border-white/10 rounded-xl p-8 text-center hover:border-cap-cyan/30 transition-colors cursor-pointer group"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={e => { e.preventDefault(); e.currentTarget.classList.add('border-cap-cyan/40') }}
              onDragLeave={e => e.currentTarget.classList.remove('border-cap-cyan/40')}
              onDrop={e => {
                e.preventDefault()
                e.currentTarget.classList.remove('border-cap-cyan/40')
                const file = e.dataTransfer.files?.[0]
                if (file) handleFileUpload(file)
              }}
            >
              <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-white/5 flex items-center justify-center group-hover:bg-cap-cyan/10 transition-colors">
                {uploading
                  ? <Loader size={22} className="text-cap-cyan animate-spin" />
                  : <Upload size={22} className="text-slate-500 group-hover:text-cap-cyan transition-colors" />
                }
              </div>
              {uploading
                ? <p className="text-sm font-medium text-cap-cyan">Uploading...</p>
                : <>
                    <p className="text-sm font-medium text-slate-300 mb-1">Drop file here or click to browse</p>
                    <p className="text-xs text-slate-500">JSON · JSONL · CSV · Parquet · ZIP (multi-file)</p>
                  </>
              }
            </div>

            {/* Conflict dialog */}
            {conflictName && (
              <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 space-y-3">
                <p className="text-sm text-amber-300 font-medium">File already exists</p>
                <p className="text-xs text-slate-400">
                  <span className="font-mono text-slate-200">{conflictName}</span> already exists in your uploads.
                  Replace it or save as a new copy?
                </p>
                <div className="flex gap-2">
                  <button onClick={handleConflictReplace} className="btn-danger text-xs px-3 py-1.5 rounded-lg">Replace</button>
                  <button onClick={handleConflictSaveNew} className="btn-secondary text-xs px-3 py-1.5 rounded-lg">Save as new</button>
                  <button onClick={() => { setPendingFile(null); setConflictName(null) }} className="text-xs text-slate-500 hover:text-slate-300 px-2">Cancel</button>
                </div>
              </div>
            )}

            {/* Previously uploaded datasets */}
            {localDatasets.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">Your Datasets</p>
                {localDatasets.map(d => (
                  <div
                    key={d.path}
                    className={cn(
                      'flex items-center gap-3 px-4 py-3 rounded-xl border text-sm transition-all',
                      selected === d.path || selected === d.name
                        ? 'bg-cap-cyan/10 border-cap-cyan/30 text-cap-cyan'
                        : 'bg-slate-800/30 border-white/[0.06] text-slate-300 hover:border-white/15',
                    )}
                  >
                    <button
                      className="flex items-center gap-3 flex-1 min-w-0 text-left"
                      onClick={() => {
                        if (d.isFolder && d.baseDir) patch({ datasetBaseDir: d.baseDir })
                        else patch({ datasetBaseDir: '' })
                        selectDataset(d.path)
                      }}
                    >
                      {d.isFolder
                        ? <FolderOpen size={14} className="shrink-0 text-amber-400" />
                        : <FileText size={14} className="shrink-0" />
                      }
                      <span className="flex-1 font-mono truncate">{d.name}</span>
                      {d.type === 'vlm_folder' && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-violet-500/15 border border-violet-500/25 text-violet-300 shrink-0">VLM</span>
                      )}
                      {d.rows && <span className="text-xs text-slate-500 shrink-0">{d.rows.toLocaleString()} rows</span>}
                    </button>
                    <button
                      onClick={e => { e.stopPropagation(); handleDelete(d.name) }}
                      className="shrink-0 p-1 rounded text-slate-600 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                      title="Delete dataset"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── Split + Format ── */}
        {selected && (
          <div className="glass-card p-5 space-y-4">
            <h2 className="text-sm font-semibold text-slate-300">Configuration</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-slate-500 mb-1.5 flex items-center gap-1.5">
                  Split
                  <InfoTooltip text="'train' = the data your model learns from." size={12} />
                </label>
                <div className="relative">
                  <select
                    value={datasetSplit}
                    onChange={e => patch({ datasetSplit: e.target.value })}
                    className="glass-input text-sm appearance-none pr-8 py-2.5"
                  >
                    {SPLITS.map(s => <option key={s} value={s}>{s}</option>)}
                  </select>
                  <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
                </div>
              </div>

              <div>
                <label className="block text-xs text-slate-500 mb-1.5 flex items-center gap-1.5">
                  Format
                  <InfoTooltip text="'auto' detects format automatically. 'custom' lets you map any columns directly — the model's own chat template is applied without any format conversion." size={12} />
                </label>
                <div className="relative">
                  <select
                    value={formatType}
                    onChange={e => patch({ formatType: e.target.value as typeof FORMATS[number] })}
                    className="glass-input text-sm appearance-none pr-8 py-2.5"
                  >
                    {FORMATS.map(f => (
                      <option key={f} value={f}>
                        {f === 'custom' ? 'custom (column mapping)' : f}
                      </option>
                    ))}
                  </select>
                  <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
                </div>
                {formatType === 'custom' && (
                  <p className="mt-1.5 text-[11px] text-cap-cyan/70 leading-relaxed">
                    Assign Input and Output columns below. The model's chat template wraps them directly — no alpaca/sharegpt conversion.
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ── Dataset Preview ── */}
        {selected && (
          <div className="glass-card p-5">
            <div className="flex items-center gap-2 mb-4">
              <Table size={15} className="text-slate-400" />
              <h2 className="text-sm font-semibold text-slate-300">Dataset Preview</h2>
              {!formatResult && !checking && (
                <button
                  onClick={() => checkFormat(selected)}
                  className="ml-auto text-xs text-cap-cyan hover:text-cap-cyan-light transition-colors"
                >
                  Inspect →
                </button>
              )}
            </div>

            {checking && (
              <div className="flex items-center gap-2 text-sm text-slate-400 py-4">
                <Loader size={15} className="animate-spin text-cap-cyan" />
                Detecting format...
              </div>
            )}

            {formatResult && (
              <div className="space-y-4">
                {/* Format badge + column pills */}
                <div className="flex items-center gap-3 flex-wrap">
                  <span className="badge-success">
                    <CheckCircle size={11} />
                    {formatResult.format} format
                  </span>
                  {(formatResult.columns ?? []).map(col => (
                    <span key={col} className="badge-neutral">{col}</span>
                  ))}
                </div>

                {/* ── Mapping status card ── */}
                {showMapping && (
                  <MappingCard
                    mapping={localMapping}
                    autoDetected={autoDetectedRef.current}
                    isVlm={isVlmMode || !!formatResult.is_image}
                  />
                )}

                {/* Path-only image warning */}
                {formatResult.image_path_only && (
                  <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 px-4 py-2.5 text-xs text-amber-300 mb-3 flex items-start gap-2">
                    <AlertCircle size={13} className="shrink-0 mt-0.5" />
                    <span>{formatResult.warning}</span>
                  </div>
                )}

                {/* Preview table — column headers have role dropdowns when mapping is active */}
                {(formatResult.columns ?? []).length > 0 && (formatResult.preview_rows ?? []).length > 0 && (
                  <div className="overflow-x-auto rounded-xl border border-white/[0.06]">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-white/[0.06] bg-slate-800/50">
                          {(formatResult.columns ?? []).map(col => (
                            <th key={col} className="text-left px-4 py-3 font-medium align-top min-w-[140px]">
                              <span className="text-[13px] font-semibold text-slate-300">{col}</span>
                              {showMapping && (
                                <HeaderRolePicker
                                  col={col}
                                  currentRole={localMapping[col]}
                                  onRoleChange={handleRoleChange}
                                  locked={localMapping[col] === 'image'}
                                />
                              )}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {(formatResult.preview_rows ?? []).map((row, i) => (
                          <tr key={i} className="border-b border-white/[0.04] hover:bg-white/[0.02]">
                            {(formatResult.columns ?? []).map(col => (
                              <td key={col} className="px-4 py-2.5 text-slate-300 max-w-xs align-top">
                                <p className="line-clamp-3 text-[12px] leading-relaxed">
                                  {(() => {
                                    const v = row[col]
                                    // image_path struct: {type: "image_path", path: "..."}
                                    if (v && typeof v === 'object' && (v as Record<string,unknown>).type === 'image_path') {
                                      return (
                                        <span className="text-amber-400/80 font-mono text-[11px]">
                                          📁 {String((v as Record<string,unknown>).path ?? '')}
                                        </span>
                                      )
                                    }
                                    return String(v ?? '')
                                  })()}
                                </p>
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            {!formatResult && !checking && selected && (
              <button
                onClick={() => checkFormat(selected)}
                className="w-full py-3 text-sm text-slate-500 hover:text-slate-300 transition-colors flex items-center justify-center gap-2"
              >
                <Search size={14} />
                Click to inspect dataset
              </button>
            )}
          </div>
        )}

        {/* ── System Prompt ── */}
        {selected && (
          <div className="glass-card p-5 space-y-3">
            <div className="flex items-center gap-2">
              <Sparkles size={14} className="text-cap-cyan" />
              <h2 className="text-sm font-semibold text-slate-300">System Prompt</h2>
              <InfoTooltip
                text="Injected as the first message in every training example. Leave empty if your dataset already has a system column or if the model was trained without one."
                size={12}
              />
              <span className="ml-auto text-[11px] text-slate-600">optional</span>
            </div>
            <textarea
              value={systemPrompt}
              onChange={e => patch({ systemPrompt: e.target.value })}
              placeholder="e.g. You are an expert legal analyst. Classify Terms of Service clauses as fair or unfair..."
              rows={3}
              className="glass-input text-sm w-full resize-none leading-relaxed"
            />
            <p className="text-[11px] text-slate-600 leading-relaxed">
              This prompt tells the model its role and task. It's prepended to every training example as a system message.
              For classification tasks, include the label categories here.
            </p>
          </div>
        )}

      </div>
    </WizardShell>
  )
}
