import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { authApi } from '@/api/auth-api'

export interface User {
  id?: number
  email: string
  name?: string
  role?: string
}

interface AuthState {
  user: User | null
  accessToken: string | null
  isAuthenticated: boolean
  isLoading: boolean

  login: (email: string, password: string) => Promise<void>
  signup: (email: string, password: string, name: string) => Promise<void>
  logout: () => void
  setTokens: (access: string, user: User) => void
  checkAuth: () => Promise<void>
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      accessToken: null,
      isAuthenticated: false,
      isLoading: false,

      login: async (email: string, password: string) => {
        set({ isLoading: true })
        try {
          const res = await authApi.login(email, password)
          set({
            accessToken: res.access_token,
            isAuthenticated: true,
            isLoading: false,
            user: { id: res.user?.id, email: res.user?.email ?? email, name: res.user?.name, role: res.user?.role },
          })
        } catch (err) {
          set({ isLoading: false })
          throw err
        }
      },

      signup: async (email: string, password: string, name: string) => {
        set({ isLoading: true })
        try {
          const res = await authApi.signup(email, password, name)
          set({
            accessToken: res.access_token,
            isAuthenticated: true,
            isLoading: false,
            user: { id: res.user?.id, email: res.user?.email ?? email, name: res.user?.name, role: res.user?.role },
          })
        } catch (err) {
          set({ isLoading: false })
          throw err
        }
      },

      logout: () => {
        set({ user: null, accessToken: null, isAuthenticated: false })
      },

      setTokens: (access: string, user: User) => {
        set({ accessToken: access, isAuthenticated: true, user })
      },

      checkAuth: async () => {
        const { accessToken } = get()
        if (!accessToken) { set({ isAuthenticated: false, user: null }); return }
        try {
          const me = await authApi.me()
          set({ isAuthenticated: true, user: { id: me.id, email: me.email, name: me.name, role: me.role } })
        } catch {
          set({ isAuthenticated: false, user: null, accessToken: null })
        }
      },
    }),
    {
      name: 'unslothcraft-auth',
      partialize: (s) => ({
        user: s.user,
        accessToken: s.accessToken,
        isAuthenticated: s.isAuthenticated,
      }),
    },
  ),
)
