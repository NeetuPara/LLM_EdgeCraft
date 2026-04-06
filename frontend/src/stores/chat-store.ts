import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface InferenceParams {
  temperature: number   // 0 = greedy (deterministic), >0 = sampling
  topP: number
  topK: number
  minP: number
  maxTokens: number
  repetitionPenalty: number
  systemPrompt: string  // kept for API use; not shown in UI for fine-tuned models
}

export const DEFAULT_PARAMS: InferenceParams = {
  temperature: 0.7,    // >0 = sampling; set to 0 for greedy/deterministic
  topP: 0.9,
  topK: 40,
  minP: 0.05,
  maxTokens: 512,
  repetitionPenalty: 1.1,
  systemPrompt: '',
}

interface ChatState {
  // Model state
  loadedModel: string | null
  loadedLora: string | null
  isLoadingModel: boolean
  loadingStatus: string   // e.g. "Loading base model...", "Applying adapter..."

  // Thread
  activeThreadId: string | null

  // Params
  params: InferenceParams

  // UI state
  chatModelType: 'text' | 'vision'
  compareMode: boolean
  compareModel: string | null
  isStreaming: boolean
  contextUsage: { prompt: number; completion: number; total: number }

  // Actions
  setLoadedModel: (model: string | null) => void
  setLoadedLora: (lora: string | null) => void
  setIsLoadingModel: (v: boolean) => void
  setLoadingStatus: (s: string) => void
  setActiveThreadId: (id: string | null) => void
  setParams: (p: Partial<InferenceParams>) => void
  setChatModelType: (t: 'text' | 'vision') => void
  setCompareMode: (v: boolean) => void
  setCompareModel: (model: string | null) => void
  setIsStreaming: (v: boolean) => void
  setContextUsage: (u: { prompt: number; completion: number; total: number }) => void
}

export const useChatStore = create<ChatState>()(
  persist(
    (set) => ({
      loadedModel: null,
      loadedLora: null,
      isLoadingModel: false,
      loadingStatus: '',
      activeThreadId: null,
      params: DEFAULT_PARAMS,
      chatModelType: 'text',
      compareMode: false,
      compareModel: null,
      isStreaming: false,
      contextUsage: { prompt: 0, completion: 0, total: 2048 },

      setLoadedModel: (model) => set({ loadedModel: model }),
      setLoadedLora: (lora) => set({ loadedLora: lora }),
      setIsLoadingModel: (v) => set({ isLoadingModel: v }),
      setLoadingStatus: (s) => set({ loadingStatus: s }),
      setActiveThreadId: (id) => set({ activeThreadId: id }),
      setParams: (p) => set((s) => ({ params: { ...s.params, ...p } })),
      setChatModelType: (t) => set({ chatModelType: t }),
      setCompareMode: (v) => set({ compareMode: v }),
      setCompareModel: (model) => set({ compareModel: model }),
      setIsStreaming: (v) => set({ isStreaming: v }),
      setContextUsage: (u) => set({ contextUsage: u }),
    }),
    {
      name: 'unslothcraft-chat',
      partialize: (s) => ({
        loadedModel: s.loadedModel,
        loadedLora: s.loadedLora,
        params: s.params,
        chatModelType: s.chatModelType,
        compareMode: s.compareMode,
        compareModel: s.compareModel,
      }),
    },
  ),
)
