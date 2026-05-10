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

export function getAgentPrompt(): Promise<{ prompt: string }> {
  return requestJson('/agent/prompt')
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

export type AgentRespondPayload = {
  instruction: string
  detections?: Record<string, unknown>[]
  audio_transcript?: string
  audio_direction?: string
  audio_distance_m?: number
  image_base64?: string
  max_new_tokens?: number
  temperature?: number
  top_p?: number
}

export type AgentRespondResult = {
  text: string
  json?: Record<string, unknown> | null
}

export function agentRespond(
  payload: AgentRespondPayload,
  signal?: AbortSignal,
): Promise<AgentRespondResult> {
  return requestJson('/agent/respond', {
    method: 'POST',
    body: JSON.stringify(payload),
    signal,
  })
}

export type AgentState = {
  running: boolean
  paused: boolean
  last_action: string
  last_reasoning: string
}

export function getAgentState(): Promise<AgentState> {
  return requestJson('/agent/state')
}
