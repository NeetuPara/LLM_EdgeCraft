// Auth API — connects to EdgeCraft backend (/api/auth/*)
// Uses email/password (not username like Unsloth Studio)
import { apiFetch } from './client'
import { isMockMode, mockAuth } from './mock'

export interface AuthStatusResponse {
  initialized: boolean
  must_change_password?: boolean
  username?: string
  email?: string
}

export interface AuthUser {
  id: number
  email: string
  name: string
  role: string
}

export interface LoginResponse {
  msg: string
  access_token: string
  token_type: string
  user: AuthUser
}

export interface SignupResponse {
  msg: string
  access_token: string
  token_type: string
  user: AuthUser
}

export const authApi = {
  // Health / status check (unauthenticated)
  status: async (): Promise<AuthStatusResponse> => {
    if (isMockMode()) return mockAuth.status()
    // Our backend doesn't have /api/auth/status — use /api/health
    const health = await apiFetch<{ status: string; device_type: string }>('/api/health', {}, true)
    return { initialized: health.status === 'ok' }
  },

  signup: (email: string, password: string, name: string) =>
    isMockMode()
      ? mockAuth.login(email, password).then(t => ({
          ...t, user: { id: 1, email, name, role: 'user' }, msg: 'Account created',
        }))
      : apiFetch<SignupResponse>('/api/auth/signup', {
          method: 'POST',
          body: JSON.stringify({ email, password, name }),
        }, true),

  login: (email: string, password: string) =>
    isMockMode()
      ? mockAuth.login(email, password).then(t => ({
          ...t, user: { id: 1, email, name: 'Demo User', role: 'user' }, msg: 'Logged in',
        }))
      : apiFetch<LoginResponse>('/api/auth/login', {
          method: 'POST',
          body: JSON.stringify({ email, password }),
        }, true),

  me: () =>
    apiFetch<AuthUser>('/api/auth/me'),

  logout: () =>
    apiFetch<void>('/api/auth/logout', { method: 'POST' }),

  changePassword: (currentPassword: string, newPassword: string) =>
    apiFetch<void>('/api/auth/change-password', {
      method: 'POST',
      body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
    }),
}
