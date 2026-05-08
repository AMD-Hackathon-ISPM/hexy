const DEFAULT_BASE_URL = ''

const baseUrl = import.meta.env.VITE_HEXY_BE_URL ?? DEFAULT_BASE_URL

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })

  if (!response.ok) {
    const message = await response.text()
    throw new Error(message || `Request failed with ${response.status}`)
  }

  return (await response.json()) as T
}

export type MujocoState = {
  time: number
  qpos: number[] | Float64Array
  qvel: number[] | Float64Array
  ctrl: number[] | Float64Array
  ncon: number
  nu: number
  njnt: number
  updated_at: number
}

export function getHealth(): Promise<{ status: string }> {
  return requestJson('/health')
}

export function getMujocoState(): Promise<MujocoState> {
  return requestJson('/mujoco/state')
}

export function resetMujoco(): Promise<MujocoState> {
  return requestJson('/mujoco/reset', { method: 'POST' })
}

export function stepMujoco(payload?: {
  ctrl?: number[]
  n_steps?: number
  key?: 'w' | 'a' | 's' | 'd'
}): Promise<MujocoState> {
  return requestJson('/mujoco/step', {
    method: 'POST',
    body: JSON.stringify(payload ?? {}),
  })
}
