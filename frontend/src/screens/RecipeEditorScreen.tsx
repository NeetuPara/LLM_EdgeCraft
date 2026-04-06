import { useState, useCallback, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  ReactFlow, Background, Controls, MiniMap,
  addEdge, useNodesState, useEdgesState,
  type NodeTypes, type Connection, Handle, Position,
  type NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import {
  ArrowLeft, Save, Play, StopCircle, Plus, Database,
  Bot, GitBranch, CheckSquare, Code2, StickyNote, Loader,
  BarChart2, Table as TableIcon, ChevronRight,
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import NavBar from '@/components/NavBar'
import { recipesDb, saveRecipe } from '@/db/recipes-db'
import { cn } from '@/utils/cn'

// ── Node type definitions ──
const nodeColors: Record<string, string> = {
  seed: '#00A5D9',
  llm: '#10B981',
  expression: '#6366F1',
  validator: '#F59E0B',
  output: '#E2E8F0',
  note: '#475569',
}

function NodeShell({ label, icon: Icon, color, children, selected }: {
  label: string; icon: React.ElementType; color: string
  children?: React.ReactNode; selected?: boolean
}) {
  return (
    <div className={cn(
      'min-w-[160px] rounded-xl border backdrop-blur-md shadow-lg transition-all',
      selected
        ? 'border-white/40 shadow-white/10'
        : 'border-white/[0.12] shadow-black/30',
    )}
      style={{ background: 'rgba(15,23,42,0.85)' }}
    >
      <div className="flex items-center gap-2 px-3 py-2.5 border-b border-white/[0.08] rounded-t-xl"
        style={{ borderTop: `3px solid ${color}` }}
      >
        <div className="p-1 rounded" style={{ background: color + '22' }}>
          <Icon size={12} style={{ color }} />
        </div>
        <span className="text-xs font-semibold text-slate-200">{label}</span>
      </div>
      {children && (
        <div className="px-3 py-2 text-[10px] text-slate-500 max-w-[200px]">
          {children}
        </div>
      )}
    </div>
  )
}

// ── Seed node ──
function SeedNode({ data, selected }: NodeProps) {
  const d = data as { label?: string; dataset?: string; file?: string }
  return (
    <>
      <NodeShell label={d.label || 'Seed'} icon={Database} color={nodeColors.seed} selected={selected}>
        {(d.dataset || d.file) && (
          <p className="truncate text-slate-400">{d.dataset || d.file}</p>
        )}
      </NodeShell>
      <Handle type="source" position={Position.Right} style={{ background: nodeColors.seed, border: 'none', width: 10, height: 10 }} />
    </>
  )
}

// ── LLM node ──
function LlmNode({ data, selected }: NodeProps) {
  const d = data as { label?: string; model?: string; prompt?: string; isVision?: boolean }
  return (
    <>
      <Handle type="target" position={Position.Left} style={{ background: nodeColors.llm, border: 'none', width: 10, height: 10 }} />
      <NodeShell label={d.label || 'LLM'} icon={Bot} color={nodeColors.llm} selected={selected}>
        {d.model && <p className="truncate text-emerald-400/80 font-mono">{d.model}</p>}
        {d.prompt && <p className="truncate text-slate-500 mt-0.5">{d.prompt?.slice(0, 40)}...</p>}
      </NodeShell>
      <Handle type="source" position={Position.Right} style={{ background: nodeColors.llm, border: 'none', width: 10, height: 10 }} />
    </>
  )
}

// ── Expression node ──
function ExpressionNode({ data, selected }: NodeProps) {
  const d = data as { label?: string; expr?: string }
  return (
    <>
      <Handle type="target" position={Position.Left} style={{ background: nodeColors.expression, border: 'none', width: 10, height: 10 }} />
      <NodeShell label={d.label || 'Expression'} icon={Code2} color={nodeColors.expression} selected={selected}>
        {d.expr && <p className="truncate text-indigo-400/80 font-mono">{d.expr?.slice(0, 35)}...</p>}
      </NodeShell>
      <Handle type="source" position={Position.Right} style={{ background: nodeColors.expression, border: 'none', width: 10, height: 10 }} />
    </>
  )
}

// ── Validator node ──
function ValidatorNode({ data, selected }: NodeProps) {
  const d = data as { label?: string }
  return (
    <>
      <Handle type="target" position={Position.Left} style={{ background: nodeColors.validator, border: 'none', width: 10, height: 10 }} />
      <NodeShell label={d.label || 'Validator'} icon={CheckSquare} color={nodeColors.validator} selected={selected} />
      <Handle type="source" position={Position.Right} style={{ background: nodeColors.validator, border: 'none', width: 10, height: 10 }} />
    </>
  )
}

// ── Output node ──
function OutputNode({ data, selected }: NodeProps) {
  const d = data as { label?: string; rows?: number }
  return (
    <>
      <Handle type="target" position={Position.Left} style={{ background: '#10B981', border: 'none', width: 10, height: 10 }} />
      <NodeShell label={d.label || 'Output'} icon={TableIcon} color="#10B981" selected={selected}>
        {d.rows && <p className="text-emerald-400">{d.rows.toLocaleString()} rows</p>}
      </NodeShell>
    </>
  )
}

// ── Note node ──
function NoteNode({ data }: NodeProps) {
  const d = data as { text?: string }
  return (
    <div className="w-40 min-h-[60px] bg-amber-500/10 border border-amber-500/20 rounded-xl p-3 text-xs text-amber-200/80 italic">
      {d.text || 'Double-click to edit'}
    </div>
  )
}

const nodeTypes: NodeTypes = {
  seed: SeedNode,
  llm: LlmNode,
  expression: ExpressionNode,
  validator: ValidatorNode,
  output: OutputNode,
  note: NoteNode,
}

// ── Block palette ──
const BLOCK_PALETTE = [
  { type: 'seed',       label: 'Seed',       icon: Database,    desc: 'HF dataset or file upload' },
  { type: 'llm',        label: 'LLM',        icon: Bot,         desc: 'Language model transform' },
  { type: 'expression', label: 'Expression', icon: Code2,       desc: 'Jinja2 template expression' },
  { type: 'validator',  label: 'Validator',  icon: CheckSquare, desc: 'Python / SQL validator' },
  { type: 'output',     label: 'Output',     icon: TableIcon,   desc: 'Final dataset output' },
  { type: 'note',       label: 'Note',       icon: StickyNote,  desc: 'Documentation note' },
]

type RunStatus = 'idle' | 'running' | 'done' | 'error'

export default function RecipeEditorScreen() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const [recipeName, setRecipeName] = useState('Untitled Recipe')
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [nodes, setNodes, onNodesChange] = useNodesState<any>([])
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [edges, setEdges, onEdgesChange] = useEdgesState<any>([])
  const [runStatus, setRunStatus] = useState<RunStatus>('idle')
  const [runRows, setRunRows] = useState(0)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [saving, setSaving] = useState(false)
  const [runProgress, setRunProgress] = useState(0)
  const [previewRows, setPreviewRows] = useState<Record<string, string>[]>([])

  // Load recipe from Dexie
  useEffect(() => {
    if (!id) return
    recipesDb.recipes.get(id).then(recipe => {
      if (!recipe) return
      setRecipeName(recipe.name)
      setNodes((recipe.nodes as Parameters<typeof setNodes>[0]) || [])
      setEdges((recipe.edges as Parameters<typeof setEdges>[0]) || [])
    })
  }, [id, setNodes, setEdges])

  const onConnect = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (params: Connection) => setEdges((eds: any[]) => addEdge({ ...params, style: { stroke: '#334155', strokeWidth: 2 } }, eds)),
    [setEdges],
  )

  const handleSave = async () => {
    if (!id) return
    setSaving(true)
    await saveRecipe(id, { name: recipeName, nodes, edges })
    setTimeout(() => setSaving(false), 600)
  }

  const addBlock = (type: string) => {
    const newNode = {
      id: `${type}-${Date.now()}`,
      type,
      position: { x: 200 + Math.random() * 200, y: 150 + Math.random() * 100 },
      data: {
        label: BLOCK_PALETTE.find(b => b.type === type)?.label ?? type,
      },
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    setNodes((nds: any[]) => [...nds, newNode])
    setPaletteOpen(false)
  }

  const handleRun = async (rows = 10) => {
    setRunStatus('running')
    setRunProgress(0)
    setPreviewRows([])

    // Simulate run
    const steps = [10, 25, 40, 60, 75, 90, 100]
    for (const pct of steps) {
      await new Promise(r => setTimeout(r, 400 + Math.random() * 300))
      setRunProgress(pct)
    }

    // Generate mock output rows
    const mockRows = Array.from({ length: rows }, (_, i) => ({
      instruction: `Sample instruction #${i + 1}: Explain the concept of ${['neural networks', 'transformers', 'backpropagation', 'attention', 'fine-tuning'][i % 5]}`,
      output: `This is a synthesized response for sample #${i + 1}. In production, this would be generated by your configured LLM node using the prompt template.`,
    }))

    setPreviewRows(mockRows)
    setRunRows(rows)
    setRunStatus('done')
  }

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-[#06090f]">
      {/* ── Editor header (no NavBar to maximize canvas space) ── */}
      <div className="shrink-0 flex items-center gap-3 px-4 py-3 border-b border-white/[0.06] bg-slate-900/60 backdrop-blur-xl z-20">
        <button
          onClick={() => { handleSave(); navigate('/recipes') }}
          className="flex items-center gap-1.5 text-slate-400 hover:text-slate-200 transition-colors text-sm"
        >
          <ArrowLeft size={16} />
          Recipes
        </button>

        <div className="w-px h-5 bg-white/10" />

        <input
          value={recipeName}
          onChange={e => setRecipeName(e.target.value)}
          className="flex-1 max-w-xs bg-transparent text-slate-200 text-sm font-semibold focus:outline-none border-b border-transparent focus:border-cap-cyan/50 transition-colors pb-0.5"
        />

        <div className="flex items-center gap-2 ml-auto">
          {/* Add block */}
          <div className="relative">
            <button
              onClick={() => setPaletteOpen(!paletteOpen)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/60 border border-white/[0.08] text-slate-300 text-sm hover:border-cap-cyan/30 transition-colors"
            >
              <Plus size={14} />
              Add Block
            </button>
            <AnimatePresence>
              {paletteOpen && (
                <motion.div
                  initial={{ opacity: 0, y: 4, scale: 0.97 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 4, scale: 0.97 }}
                  transition={{ duration: 0.12 }}
                  className="absolute top-full right-0 mt-2 w-56 bg-slate-900/95 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl z-50 overflow-hidden py-1"
                >
                  {BLOCK_PALETTE.map(({ type, label, icon: Icon, desc }) => (
                    <button
                      key={type}
                      onClick={() => addBlock(type)}
                      className="w-full flex items-center gap-3 px-4 py-2.5 hover:bg-white/5 transition-colors text-sm"
                    >
                      <div className="p-1.5 rounded-lg" style={{ background: (nodeColors[type] ?? '#475569') + '22' }}>
                        <Icon size={13} style={{ color: nodeColors[type] ?? '#475569' }} />
                      </div>
                      <div className="text-left">
                        <p className="text-slate-200 font-medium">{label}</p>
                        <p className="text-[10px] text-slate-500">{desc}</p>
                      </div>
                    </button>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Save */}
          <button onClick={handleSave} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg btn-secondary text-sm py-1.5">
            {saving ? <Loader size={13} className="animate-spin" /> : <Save size={13} />}
            Save
          </button>

          {/* Run */}
          <button
            onClick={() => runStatus === 'running' ? setRunStatus('idle') : handleRun(10)}
            className={cn(
              'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
              runStatus === 'running'
                ? 'bg-red-500/20 border border-red-500/30 text-red-400 hover:bg-red-500/30'
                : 'bg-emerald-500/20 border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/30',
            )}
          >
            {runStatus === 'running'
              ? <><StopCircle size={13} /> Stop</>
              : <><Play size={13} /> Preview (10 rows)</>
            }
          </button>
        </div>
      </div>

      {/* ── Canvas ── */}
      <div className="flex-1 relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.3 }}
          style={{ background: 'transparent' }}
          defaultEdgeOptions={{
            style: { stroke: '#334155', strokeWidth: 2 },
            animated: runStatus === 'running',
          }}
        >
          <Background color="#1E293B" gap={24} size={1} />
          <Controls
            style={{
              background: 'rgba(15,23,42,0.9)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: '12px',
            }}
          />
          <MiniMap
            style={{
              background: 'rgba(15,23,42,0.9)',
              border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: '12px',
            }}
            nodeColor={(n) => nodeColors[n.type ?? ''] ?? '#334155'}
          />
        </ReactFlow>

        {/* Empty canvas hint */}
        {nodes.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-cap-cyan/10 border border-cap-cyan/20 flex items-center justify-center">
                <GitBranch size={28} className="text-cap-cyan" />
              </div>
              <p className="text-slate-400 font-medium mb-2">Canvas is empty</p>
              <p className="text-slate-600 text-sm">Click "Add Block" to add nodes to your recipe</p>
            </div>
          </div>
        )}

        {/* Run progress overlay */}
        <AnimatePresence>
          {runStatus === 'running' && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 8 }}
              className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-3 px-5 py-3 bg-slate-900/95 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl z-10"
            >
              <Loader size={15} className="animate-spin text-cap-cyan" />
              <div>
                <p className="text-sm font-medium text-slate-200">Generating dataset...</p>
                <div className="flex items-center gap-2 mt-1">
                  <div className="w-32 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-cap-cyan rounded-full"
                      animate={{ width: `${runProgress}%` }}
                      transition={{ duration: 0.4 }}
                    />
                  </div>
                  <span className="text-xs text-slate-500">{runProgress}%</span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── Results panel ── */}
      <AnimatePresence>
        {runStatus === 'done' && previewRows.length > 0 && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 280, opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="shrink-0 border-t border-white/[0.06] bg-slate-900/80 backdrop-blur-xl overflow-hidden"
          >
            <div className="flex items-center justify-between px-5 py-3 border-b border-white/[0.06]">
              <div className="flex items-center gap-2">
                <BarChart2 size={14} className="text-emerald-400" />
                <span className="text-sm font-medium text-slate-200">
                  Preview — {runRows} rows generated
                </span>
                <span className="badge-success text-xs">done</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleRun(100)}
                  className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20 transition-colors"
                >
                  <Play size={11} />
                  Full Run (100 rows)
                </button>
                <button
                  onClick={() => setRunStatus('idle')}
                  className="text-slate-500 hover:text-slate-300 transition-colors p-1"
                >
                  ×
                </button>
              </div>
            </div>
            <div className="overflow-auto h-[calc(100%-49px)]">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-white/[0.06] bg-slate-800/30 sticky top-0">
                    {Object.keys(previewRows[0] ?? {}).map(col => (
                      <th key={col} className="text-left px-5 py-2.5 text-slate-500 font-medium capitalize">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewRows.slice(0, 8).map((row, i) => (
                    <tr key={i} className="border-b border-white/[0.03] hover:bg-white/[0.02]">
                      {Object.values(row).map((val, j) => (
                        <td key={j} className="px-5 py-2.5 text-slate-300 max-w-[300px]">
                          <span className="line-clamp-2">{String(val)}</span>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              {previewRows.length > 8 && (
                <p className="text-center text-xs text-slate-600 py-3">
                  Showing 8 of {previewRows.length} rows
                </p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
