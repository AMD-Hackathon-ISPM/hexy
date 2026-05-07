import { useEffect, useRef } from 'react'
import { useRobotStatusStore } from '@/stores/useRobotStatusStore'

const DEFAULT_BASE_URL = ''
const STREAM_INTERVAL_SEC = 3
const ALERT_TTL_MS = 4500

function buildWsUrl(baseUrl: string) {
  if (!baseUrl) {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    return `${proto}://${window.location.host}/audio/whisper?interval_sec=${STREAM_INTERVAL_SEC}`
  }
  const wsBase = baseUrl.replace(/^http/, 'ws')
  return `${wsBase}/audio/whisper?interval_sec=${STREAM_INTERVAL_SEC}`
}

type WhisperTranscriptMessage = {
  transcript?: string
  timestamp?: number
  direction?: string
  pan?: number
}

type WhisperAlertMessage = {
  audio_alert?: boolean
  keyword?: string
}

export function useWhisperStream() {
  const setAudioTranscript = useRobotStatusStore((s) => s.setAudioTranscript)
  const setAudioAlert = useRobotStatusStore((s) => s.setAudioAlert)
  const reconnectTimerRef = useRef<number | null>(null)
  const alertTimerRef = useRef<number | null>(null)
  const socketRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let cancelled = false

    const clearReconnectTimer = () => {
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }
    }

    const clearAlertTimer = () => {
      if (alertTimerRef.current !== null) {
        window.clearTimeout(alertTimerRef.current)
        alertTimerRef.current = null
      }
    }

    const scheduleReconnect = () => {
      if (cancelled || reconnectTimerRef.current !== null) return
      reconnectTimerRef.current = window.setTimeout(() => {
        reconnectTimerRef.current = null
        connect()
      }, 1500)
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
        if (import.meta.env.DEV) {
          console.info('[whisper] ws open', wsUrl)
        }
      }

      socket.onmessage = (event) => {
        if (cancelled) return
        try {
          const payload = JSON.parse(event.data) as WhisperTranscriptMessage & WhisperAlertMessage

          if (payload.transcript) {
            setAudioTranscript({
              text: payload.transcript,
              timestamp: payload.timestamp ?? Date.now() / 1000,
              direction: payload.direction,
              pan: payload.pan,
            })
          }

          if (payload.audio_alert && payload.keyword) {
            setAudioAlert({ keyword: payload.keyword, timestamp: Date.now() })
            clearAlertTimer()
            alertTimerRef.current = window.setTimeout(() => {
              setAudioAlert(undefined)
              alertTimerRef.current = null
            }, ALERT_TTL_MS)
          }
        } catch {
          // Ignore malformed messages.
        }
      }

      const handleClose = () => {
        if (import.meta.env.DEV) {
          console.info('[whisper] ws closed')
        }
        if (socketRef.current === socket) {
          socketRef.current = null
        }
        scheduleReconnect()
      }

      socket.onerror = () => {
        if (import.meta.env.DEV) {
          console.info('[whisper] ws error')
        }
        handleClose()
      }
      socket.onclose = () => {
        handleClose()
      }
    }

    const connectTimer = window.setTimeout(() => connect(), 0)

    return () => {
      cancelled = true
      window.clearTimeout(connectTimer)
      clearReconnectTimer()
      clearAlertTimer()

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
  }, [setAudioAlert, setAudioTranscript])
}
