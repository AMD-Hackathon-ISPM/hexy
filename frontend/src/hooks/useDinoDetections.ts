import { useEffect, useRef } from 'react'
import { useRobotStatusStore } from '@/stores/useRobotStatusStore'
import { useViewportStore } from '@/stores/useViewportStore'
import type { DinoDetection } from '@/stores/useRobotStatusStore'

const DEFAULT_BASE_URL = ''
const STREAM_INTERVAL_MS = 200
const FRAME_WIDTH = 640
const FRAME_HEIGHT = 360

function buildWsUrl(baseUrl: string, camera: string) {
  const params = new URLSearchParams({
    interval_ms: String(STREAM_INTERVAL_MS),
    width: String(FRAME_WIDTH),
    height: String(FRAME_HEIGHT),
  })

  if (camera) {
    params.set('camera', camera)
  }

  if (!baseUrl) {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    return `${proto}://${window.location.host}/mujoco/detections?${params.toString()}`
  }
  const wsBase = baseUrl.replace(/^http/, 'ws')
  return `${wsBase}/mujoco/detections?${params.toString()}`
}

type DetectionPayload = {
  detections?: DinoDetection[]
}

export function useDinoDetections() {
  const setDinoDetections = useRobotStatusStore((s) => s.setDinoDetections)
  const dinoCamera = useViewportStore((s) => s.dinoCamera)
  const reconnectTimerRef = useRef<number | null>(null)
  const socketRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let cancelled = false

    const clearReconnectTimer = () => {
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
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
      const wsUrl = buildWsUrl(baseUrl, dinoCamera)
      const socket = new WebSocket(wsUrl)
      socketRef.current = socket

      socket.onopen = () => {
        if (cancelled) return
        clearReconnectTimer()
        if (import.meta.env.DEV) {
          console.info('[dino] ws open', wsUrl)
        }
      }

      socket.onmessage = (event) => {
        if (cancelled) return
        try {
          const payload = JSON.parse(event.data) as DetectionPayload
          if (!payload.detections) return
          setDinoDetections({
            detections: payload.detections,
            frameWidth: FRAME_WIDTH,
            frameHeight: FRAME_HEIGHT,
            updatedAt: Date.now(),
          })
        } catch {
          // Ignore malformed messages.
        }
      }

      const handleClose = () => {
        if (import.meta.env.DEV) {
          console.info('[dino] ws closed')
        }
        if (socketRef.current === socket) {
          socketRef.current = null
        }
        scheduleReconnect()
      }

      socket.onerror = () => {
        if (import.meta.env.DEV) {
          console.info('[dino] ws error')
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
  }, [dinoCamera, setDinoDetections])
}
