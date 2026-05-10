import { useEffect, useRef } from 'react'
import { getAgentState } from '@/lib/backendClient'
import { useRobotStatusStore } from '@/stores/useRobotStatusStore'

const ACTION_LABELS: Record<string, string> = {
  w: 'Forward',
  s: 'Backward',
  a: 'Turn Left',
  d: 'Turn Right',
  stop: 'Stopped',
  '': 'Idle',
}

export function useAgentState(intervalMs = 2000) {
  const setStatus = useRobotStatusStore((s) => s.setStatus)
  const addReasoningStep = useRobotStatusStore((s) => s.addReasoningStep)
  const prevReasoningRef = useRef('')

  useEffect(() => {
    let cancelled = false

    const poll = async () => {
      try {
        const state = await getAgentState()
        if (cancelled) return

        const agentMode = state.paused ? 'PAUSED' : state.running ? 'Autonomous' : 'Stopped'
        const agentAction = ACTION_LABELS[state.last_action] ?? state.last_action ?? 'Idle'

        setStatus({ agentMode, agentAction })

        if (
          state.last_reasoning &&
          state.last_reasoning !== prevReasoningRef.current
        ) {
          prevReasoningRef.current = state.last_reasoning
          addReasoningStep({
            text: `[${(state.last_action || '?').toUpperCase()}] ${state.last_reasoning}`,
            status: 'done',
            timestamp: Date.now(),
          })
        }
      } catch {
        // backend may not be ready yet — silently retry
      }
    }

    poll()
    const id = setInterval(poll, intervalMs)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [intervalMs, setStatus, addReasoningStep])
}
