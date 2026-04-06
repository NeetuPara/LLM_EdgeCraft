import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useLiveQuery } from 'dexie-react-hooks'
import {
  Plus, Trash2, FlaskConical, BookOpen, Code,
  FileText, Database, Table, Braces, Sparkles,
  ChevronRight, Clock,
} from 'lucide-react'
import { motion } from 'framer-motion'
import NavBar from '@/components/NavBar'
import { recipesDb, createRecipe, deleteRecipe } from '@/db/recipes-db'
import type { Recipe } from '@/db/recipes-db'
import { cn } from '@/utils/cn'

// ── Template definitions ──
const TEMPLATES = [
  {
    id: 'instruction-from-answer',
    name: 'Instruction from Answer',
    desc: 'Given answers/documents, auto-generate instruction questions using an LLM.',
    icon: BookOpen,
    color: 'cap-cyan',
    difficulty: 'Easy',
    nodes: [
      { id: 'seed-1', type: 'seed', position: { x: 60, y: 160 }, data: { label: 'HF Dataset', dataset: 'yahma/alpaca-cleaned' } },
      { id: 'llm-1', type: 'llm', position: { x: 320, y: 120 }, data: { label: 'Generate Question', model: 'gpt-4o-mini', prompt: 'Given this answer, write a question: {{output}}' } },
      { id: 'out-1', type: 'output', position: { x: 580, y: 160 }, data: { label: 'Output Dataset' } },
    ],
    edges: [
      { id: 'e1', source: 'seed-1', target: 'llm-1' },
      { id: 'e2', source: 'llm-1', target: 'out-1' },
    ],
  },
  {
    id: 'pdf-qa',
    name: 'PDF Q&A',
    desc: 'Extract text from PDFs and generate Q&A pairs for fine-tuning.',
    icon: FileText,
    color: 'amber-400',
    difficulty: 'Medium',
    nodes: [
      { id: 'seed-1', type: 'seed', position: { x: 60, y: 160 }, data: { label: 'PDF Upload', file: 'document.pdf' } },
      { id: 'llm-1', type: 'llm', position: { x: 320, y: 100 }, data: { label: 'Generate Q&A', model: 'gpt-4o', prompt: 'Generate 5 Q&A pairs from: {{chunk}}' } },
      { id: 'out-1', type: 'output', position: { x: 580, y: 160 }, data: { label: 'Q&A Dataset' } },
    ],
    edges: [
      { id: 'e1', source: 'seed-1', target: 'llm-1' },
      { id: 'e2', source: 'llm-1', target: 'out-1' },
    ],
  },
  {
    id: 'text-to-python',
    name: 'Text → Python',
    desc: 'Transform natural language descriptions into Python code examples.',
    icon: Code,
    color: 'indigo-400',
    difficulty: 'Medium',
    nodes: [
      { id: 'seed-1', type: 'seed', position: { x: 60, y: 160 }, data: { label: 'Task Descriptions' } },
      { id: 'llm-1', type: 'llm', position: { x: 320, y: 120 }, data: { label: 'Write Python', model: 'gpt-4o', prompt: 'Write Python code for: {{instruction}}' } },
      { id: 'val-1', type: 'validator', position: { x: 560, y: 120 }, data: { label: 'Syntax Check' } },
      { id: 'out-1', type: 'output', position: { x: 780, y: 160 }, data: { label: 'Code Dataset' } },
    ],
    edges: [
      { id: 'e1', source: 'seed-1', target: 'llm-1' },
      { id: 'e2', source: 'llm-1', target: 'val-1' },
      { id: 'e3', source: 'val-1', target: 'out-1' },
    ],
  },
  {
    id: 'text-to-sql',
    name: 'Text → SQL',
    desc: 'Generate natural language to SQL query pairs from database schemas.',
    icon: Database,
    color: 'emerald-400',
    difficulty: 'Medium',
    nodes: [
      { id: 'seed-1', type: 'seed', position: { x: 60, y: 160 }, data: { label: 'Schema Seed' } },
      { id: 'llm-1', type: 'llm', position: { x: 320, y: 120 }, data: { label: 'Generate SQL', model: 'gpt-4o', prompt: 'Write SQL for: {{question}} using schema: {{schema}}' } },
      { id: 'val-1', type: 'validator', position: { x: 560, y: 120 }, data: { label: 'SQL Validator' } },
      { id: 'out-1', type: 'output', position: { x: 780, y: 160 }, data: { label: 'SQL Dataset' } },
    ],
    edges: [
      { id: 'e1', source: 'seed-1', target: 'llm-1' },
      { id: 'e2', source: 'llm-1', target: 'val-1' },
      { id: 'e3', source: 'val-1', target: 'out-1' },
    ],
  },
  {
    id: 'structured-outputs',
    name: 'Structured Outputs',
    desc: 'Synthesize structured JSON objects from seed data and templates.',
    icon: Braces,
    color: 'pink-400',
    difficulty: 'Hard',
    nodes: [
      { id: 'seed-1', type: 'seed', position: { x: 60, y: 160 }, data: { label: 'Seed Data' } },
      { id: 'expr-1', type: 'expression', position: { x: 280, y: 100 }, data: { label: 'Template', expr: '{"name": "{{name}}", "value": {{value}}}' } },
      { id: 'llm-1', type: 'llm', position: { x: 500, y: 120 }, data: { label: 'Enrich JSON', model: 'gpt-4o' } },
      { id: 'out-1', type: 'output', position: { x: 720, y: 160 }, data: { label: 'Structured Dataset' } },
    ],
    edges: [
      { id: 'e1', source: 'seed-1', target: 'expr-1' },
      { id: 'e2', source: 'expr-1', target: 'llm-1' },
      { id: 'e3', source: 'llm-1', target: 'out-1' },
    ],
  },
  {
    id: 'ocr',
    name: 'OCR + Vision Q&A',
    desc: 'Process images with OCR and generate visual question-answer pairs.',
    icon: Table,
    color: 'orange-400',
    difficulty: 'Hard',
    nodes: [
      { id: 'seed-1', type: 'seed', position: { x: 60, y: 160 }, data: { label: 'Image Dataset' } },
      { id: 'llm-1', type: 'llm', position: { x: 320, y: 120 }, data: { label: 'Vision LLM', model: 'gpt-4o', isVision: true } },
      { id: 'out-1', type: 'output', position: { x: 580, y: 160 }, data: { label: 'Vision Dataset' } },
    ],
    edges: [
      { id: 'e1', source: 'seed-1', target: 'llm-1' },
      { id: 'e2', source: 'llm-1', target: 'out-1' },
    ],
  },
]

const DIFF_COLORS: Record<string, string> = {
  Easy: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  Medium: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  Hard: 'text-red-400 bg-red-500/10 border-red-500/20',
}

function formatTime(ts: number): string {
  const d = new Date(ts)
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

export default function DataRecipesScreen() {
  const navigate = useNavigate()
  const [creating, setCreating] = useState(false)

  const recipes = useLiveQuery(
    () => recipesDb.recipes.orderBy('updatedAt').reverse().toArray(),
    [], [] as Recipe[],
  )

  const handleCreateFromTemplate = async (template: typeof TEMPLATES[0]) => {
    setCreating(true)
    const recipe = await createRecipe(template.name, template.nodes, template.edges)
    navigate(`/recipes/${recipe.id}`)
  }

  const handleCreateBlank = async () => {
    setCreating(true)
    const recipe = await createRecipe('Untitled Recipe')
    navigate(`/recipes/${recipe.id}`)
  }

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    await deleteRecipe(id)
  }

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <NavBar />
      <div className="flex-1 overflow-y-auto min-h-0">
        <div className="max-w-6xl mx-auto px-6 py-8">

          {/* Header */}
          <motion.div
            initial={{ opacity: 1, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-end justify-between mb-8"
          >
            <div>
              <h1 className="text-2xl font-bold text-slate-100 font-display mb-1">Data Recipes</h1>
              <p className="text-slate-400 text-sm">Visually design synthetic datasets for fine-tuning.</p>
            </div>
            <button
              onClick={handleCreateBlank}
              disabled={creating}
              className="btn-primary flex items-center gap-2"
            >
              <Plus size={17} />
              New Recipe
            </button>
          </motion.div>

          {/* ── Learning templates ── */}
          <motion.div
            initial={{ opacity: 1, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
          >
            <div className="flex items-center gap-2 mb-4">
              <Sparkles size={15} className="text-cap-cyan" />
              <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Learning Recipes</h2>
              <span className="text-xs text-slate-600">— start from a template</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-10">
              {TEMPLATES.map((template, i) => (
                <motion.div
                  key={template.id}
                  initial={{ opacity: 1, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.05 + i * 0.04 }}
                >
                  <button
                    onClick={() => handleCreateFromTemplate(template)}
                    disabled={creating}
                    className="glass-card-interactive w-full text-left h-full"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className={cn('p-2.5 rounded-xl', `bg-${template.color}/10 border border-${template.color}/20`)}>
                        <template.icon size={18} className={`text-${template.color}`} />
                      </div>
                      <span className={cn('text-[10px] font-bold px-2 py-0.5 rounded-full border', DIFF_COLORS[template.difficulty])}>
                        {template.difficulty}
                      </span>
                    </div>
                    <h3 className="font-semibold text-slate-200 text-sm mb-1">{template.name}</h3>
                    <p className="text-xs text-slate-500 leading-relaxed mb-3">{template.desc}</p>
                    <div className="flex items-center gap-1 text-xs text-slate-600">
                      <span>{template.nodes.length} nodes</span>
                      <span>·</span>
                      <span>{template.edges.length} connections</span>
                      <ChevronRight size={12} className="ml-auto text-slate-600 group-hover:text-cap-cyan transition-colors" />
                    </div>
                  </button>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* ── My recipes ── */}
          <motion.div
            initial={{ opacity: 1, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <div className="flex items-center gap-2 mb-4">
              <FlaskConical size={15} className="text-slate-400" />
              <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">My Recipes</h2>
            </div>

            {(recipes as Recipe[]).length === 0 ? (
              <div className="glass-card text-center py-12">
                <div className="w-14 h-14 mx-auto mb-4 rounded-2xl bg-white/5 flex items-center justify-center">
                  <FlaskConical size={26} className="text-slate-600" />
                </div>
                <p className="text-slate-400 font-medium mb-1">No recipes yet</p>
                <p className="text-slate-600 text-sm">Create one from a template or start blank above.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {(recipes as Recipe[]).map((recipe: Recipe, i: number) => (
                  <motion.div
                    key={recipe.id}
                    initial={{ opacity: 1, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.05 }}
                  >
                    <div
                      onClick={() => navigate(`/recipes/${recipe.id}`)}
                      className="glass-card-interactive group relative"
                    >
                      <button
                        onClick={(e) => handleDelete(recipe.id, e)}
                        className="absolute top-4 right-4 p-1.5 opacity-0 group-hover:opacity-100 text-slate-600 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-all"
                      >
                        <Trash2 size={13} />
                      </button>
                      <div className="flex items-center gap-2 mb-2">
                        <div className="p-2 rounded-lg bg-cap-cyan/10 border border-cap-cyan/20">
                          <FlaskConical size={14} className="text-cap-cyan" />
                        </div>
                        <h3 className="font-semibold text-slate-200 text-sm truncate">{recipe.name}</h3>
                      </div>
                      <div className="flex items-center gap-3 text-xs text-slate-600">
                        <span>{(recipe.nodes as unknown[]).length} nodes</span>
                        <span>·</span>
                        <span className="flex items-center gap-1">
                          <Clock size={10} />
                          {formatTime(recipe.updatedAt)}
                        </span>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </motion.div>

          <div className="h-8" />
        </div>
      </div>
    </div>
  )
}
