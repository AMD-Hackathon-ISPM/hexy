import { useEffect } from 'react'
import { getHealth, getMujocoState } from '@/lib/backendClient'
import { useRobotStatusStore } from '@/stores/useRobotStatusStore'

const POLL_INTERVAL_MS = 2000

export function useBackendStatus() {
  const setStatus = useRobotStatusStore((s) => s.setStatus)

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined

    const poll = async () => {
      try {
        await getHealth()
        if (cancelled) return
        setStatus({ connection: 'online', mode: 'MuJoCo' })

        await getMujocoState()
        if (cancelled) return
      } catch {
        if (!cancelled) {
          setStatus({ connection: 'offline' })
        }
      } finally {
        timer = window.setTimeout(poll, POLL_INTERVAL_MS)
      }
    }

    poll()

    return () => {
      cancelled = true
      if (timer) window.clearTimeout(timer)
    }
  }, [setStatus])

}
