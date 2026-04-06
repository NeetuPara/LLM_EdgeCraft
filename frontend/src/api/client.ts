// Core API fetch wrapper — handles JWT injection, auto-refresh, and error dispatch

const SESSION_EXPIRED_EVENT = 'auth:session-expired'

function getAccessToken(): string | null {
  try {
    const raw = localStorage.getItem('unslothcraft-auth')
    if (!raw) return null
    const parsed = JSON.parse(raw)
    return parsed?.state?.accessToken ?? null
  } catch {
    return null
  }
}

function getRefreshToken(): string | null {
  try {
    const raw = localStorage.getItem('unslothcraft-auth')
    if (!raw) return null
    const parsed = JSON.parse(raw)
    return parsed?.state?.refreshToken ?? null
  } catch {
    return null
  }
}

async function refreshAccessToken(): Promise<string | null> {
  const refreshToken = getRefreshToken()
  if (!refreshToken) return null

  try {
    const res = await fetch('/api/auth/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken }),
    })
    if (!res.ok) return null

    const data = await res.json()
    const newToken = data.access_token

    // Update stored tokens
    const raw = localStorage.getItem('unslothcraft-auth')
    if (raw && newToken) {
      const parsed = JSON.parse(raw)
      parsed.state.accessToken = newToken
      localStorage.setItem('unslothcraft-auth', JSON.stringify(parsed))
    }

    return newToken
  } catch {
    return null
  }
}

export interface ApiError {
  status: number
  message: string
  detail?: unknown
}

export async function apiFetch<T = unknown>(
  path: string,
  options: RequestInit = {},
  skipAuth = false,
): Promise<T> {
  const token = getAccessToken()

  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  }

  // Don't set Content-Type for FormData (browser sets it with boundary)
  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = headers['Content-Type'] ?? 'application/json'
  }

  if (token && !skipAuth) {
    headers['Authorization'] = `Bearer ${token}`
  }

  let res = await fetch(path, { ...options, headers })

  // Auto-refresh on 401 (skip for auth endpoints to avoid loops)
  if (res.status === 401 && !skipAuth && !path.includes('/api/auth/')) {
    const newToken = await refreshAccessToken()
    if (newToken) {
      headers['Authorization'] = `Bearer ${newToken}`
      res = await fetch(path, { ...options, headers })
    } else {
      // Refresh failed — session expired
      window.dispatchEvent(new CustomEvent(SESSION_EXPIRED_EVENT))
      const err: ApiError = { status: 401, message: 'Session expired' }
      throw err
    }
  }

  if (!res.ok) {
    let message = `HTTP ${res.status}`
    let detail: unknown
    try {
      const body = await res.json()
      message = body.detail ?? body.message ?? message
      detail = body
    } catch {
      // ignore parse error
    }
    const err: ApiError = { status: res.status, message: String(message), detail }
    throw err
  }

  // Handle empty responses (204 No Content, etc.)
  const contentType = res.headers.get('content-type') ?? ''
  if (!contentType.includes('application/json') || res.status === 204) {
    return undefined as T
  }

  return res.json() as Promise<T>
}

// SSE connection helper
export function createSSEConnection(
  path: string,
  onMessage: (event: MessageEvent) => void,
  onError?: (error: Event) => void,
  lastEventId?: string,
): EventSource {
  const token = getAccessToken()
  const url = new URL(path, window.location.origin)
  if (token) url.searchParams.set('token', token)
  if (lastEventId) url.searchParams.set('last_event_id', lastEventId)

  const es = new EventSource(url.toString())
  es.onmessage = onMessage
  if (onError) es.onerror = onError
  return es
}
