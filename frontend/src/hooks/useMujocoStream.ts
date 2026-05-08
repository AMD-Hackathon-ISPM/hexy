import { useEffect, useRef } from 'react'
import type { MujocoState } from '@/lib/backendClient'
import { useRobotStatusStore } from '@/stores/useRobotStatusStore'

const DEFAULT_BASE_URL = ''
const STREAM_INTERVAL_MS = 50
const UI_UPDATE_INTERVAL_MS = 50

function buildWsUrl(baseUrl: string) {
  if (!baseUrl) {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    return `${proto}://${window.location.host}/mujoco/stream?interval_ms=${STREAM_INTERVAL_MS}`
  }
  const wsBase = baseUrl.replace(/^http/, 'ws')
  return `${wsBase}/mujoco/stream?interval_ms=${STREAM_INTERVAL_MS}`
}

export function useMujocoStream() {
  const setStatus = useRobotStatusStore((s) => s.setStatus)
  const setMujocoStreamState = useRobotStatusStore((s) => s.setMujocoStreamState)
  const reconnectTimerRef = useRef<number | null>(null)
  const connectTimerRef = useRef<number | null>(null)
  const lastUiUpdateRef = useRef(0)
  const socketRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let cancelled = false

    const clearReconnectTimer = () => {
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }
    }

    const clearConnectTimer = () => {
      if (connectTimerRef.current !== null) {
        window.clearTimeout(connectTimerRef.current)
        connectTimerRef.current = null
      }
    }

    const scheduleReconnect = () => {
      if (cancelled || reconnectTimerRef.current !== null) return
      setStatus({ connection: 'offline' })
      reconnectTimerRef.current = window.setTimeout(() => {
        reconnectTimerRef.current = null
        connect()
      }, 1000)
    }

    const connect = () => {
      const existingSocket = socketRef.current
      if (
        existingSocket &&
        (existingSocket.readyState === WebSocket.OPEN ||
          existingSocket.readyState === WebSocket.CONNECTING)
      ) {
        return
      }

      const baseUrl = import.meta.env.VITE_HEXY_BE_URL ?? DEFAULT_BASE_URL
      const wsUrl = buildWsUrl(baseUrl)
      const socket = new WebSocket(wsUrl)
      socketRef.current = socket

      socket.onopen = () => {
        if (cancelled) return
        clearReconnectTimer()
        setStatus({ connection: 'online', mode: 'MuJoCo Stream' })
        if (import.meta.env.DEV) {
          console.info('[mujoco] ws open', wsUrl)
        }
      }

      socket.onmessage = (event) => {
        if (cancelled) return
        try {
          const payload = JSON.parse(event.data) as MujocoState | { detail?: string }
          if ('detail' in payload && payload.detail) return

          const state = payload as MujocoState
          const now = Date.now()
          if (now - lastUiUpdateRef.current < UI_UPDATE_INTERVAL_MS) return
          lastUiUpdateRef.current = now

          setMujocoStreamState(state)
        } catch {
          // Ignore malformed messages.
        }
      }

      const handleClose = () => {
        if (import.meta.env.DEV) {
          console.info('[mujoco] ws closed')
        }
        if (socketRef.current === socket) {
          socketRef.current = null
        }
        scheduleReconnect()
      }

      socket.onerror = () => {
        if (import.meta.env.DEV) {
          console.info('[mujoco] ws error')
        }
        handleClose()
      }
      socket.onclose = (event) => {
        if (import.meta.env.DEV) {
          console.info('[mujoco] ws close', {
            code: event.code,
            reason: event.reason,
            wasClean: event.wasClean,
          })
        }
        handleClose()
      }
    }

    // Defer the initial connect so React StrictMode's dev-only throwaway mount
    // can clean up before any socket is created.
    connectTimerRef.current = window.setTimeout(() => {
      connectTimerRef.current = null
      connect()
    }, 0)

    return () => {
      if (import.meta.env.DEV) {
        console.info('[mujoco] ws cleanup')
      }
      cancelled = true
      clearConnectTimer()
      clearReconnectTimer()

      const socket = socketRef.current
      socketRef.current = null

      if (socket) {
        socket.onerror = null
        socket.onclose = null
        socket.onopen = null
        socket.onmessage = null
      }

      if (
        socket &&
        (socket.readyState === WebSocket.OPEN ||
          socket.readyState === WebSocket.CONNECTING)
      ) {
        socket.close()
      }
    }
  }, [setMujocoStreamState, setStatus])
}
