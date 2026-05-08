import { Canvas } from '@react-three/fiber'
import { MujocoPhysics, MujocoProvider } from 'mujoco-react'
import type { SceneConfig } from 'mujoco-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useViewportStore } from '@/stores/useViewportStore'
import { SceneStage } from './SceneStage'
import { PipCameraFrame } from './PipCameraFrame'
import { CameraTransitionOverlay } from './CameraTransitionOverlay'
import RobotOverlay from './RobotOverlay'

const DEFAULT_BACKEND_URL = ''
const DEFAULT_SCENE_FILE = 'rl/models/hexapod_static.xml'
const CAVE_SCENE_FILE = 'cave/cave_hexapod.xml'
const CAVE_PREFLIGHT_TIMEOUT_MS = 6000
const CAVE_LOAD_TIMEOUT_MS = 20000
const CAVE_DISABLED_SESSION_KEY = 'hexy:cave-browser-disabled'

function buildHexyConfig(baseUrl: string, sceneFile: string): SceneConfig {
  const normalizedBase = baseUrl.endsWith('/')
    ? baseUrl.slice(0, -1)
    : baseUrl

  return {
    src: `${normalizedBase}/assets/`,
    sceneFile,
  }
}

function isAbortError(error: unknown) {
  return error instanceof DOMException && error.name === 'AbortError'
}

async function fetchCaveScene(url: string) {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), CAVE_PREFLIGHT_TIMEOUT_MS)

  try {
    return await fetch(url, {
      cache: 'force-cache',
      signal: controller.signal,
    })
  } finally {
    window.clearTimeout(timeout)
  }
}

type RobotSceneProps = {
  onMainReady?: () => void
  onLoadError?: (message: string) => void
  onLoadDetail?: (message: string) => void
  showPip?: boolean
}

export function RobotScene({
  onMainReady,
  onLoadError,
  onLoadDetail,
  showPip = true,
}: RobotSceneProps) {
  const pipSlot = useViewportStore((s) => s.pipSlot)
  const pipCollapsed = useViewportStore((s) => s.pipCollapsed)
  const viewMode = useViewportStore((s) => s.viewMode)
  const mainViewRef = useRef<HTMLDivElement | null>(null)
  const pipViewRef = useRef<HTMLDivElement | null>(null)
  const sceneReadyRef = useRef(false)
  const [sceneFile, setSceneFile] = useState<string | null>(null)
  const mainCameraPreset = pipSlot === 'robotPOV' ? 'orbit' : 'robotPOV'
  const renderPip = showPip && !pipCollapsed
  const showMainRobotOverlay = mainCameraPreset === 'robotPOV'
  const showPipRobotOverlay = mainCameraPreset !== 'robotPOV'

  useEffect(() => {
    const override = import.meta.env.VITE_HEXY_SCENE_FILE
    if (override) {
      onLoadDetail?.('Loading configured MuJoCo scene')
      setSceneFile(override)
      return
    }

    if (window.sessionStorage.getItem(CAVE_DISABLED_SESSION_KEY) === '1') {
      onLoadDetail?.('Cave scene timed out in this tab; loading static hexapod')
      setSceneFile(DEFAULT_SCENE_FILE)
      return
    }

    onLoadDetail?.('Checking for generated cave scene')
    const baseUrl = import.meta.env.VITE_HEXY_BE_URL ?? DEFAULT_BACKEND_URL
    const normalizedBase = baseUrl.endsWith('/')
      ? baseUrl.slice(0, -1)
      : baseUrl
    let cancelled = false

    fetchCaveScene(`${normalizedBase}/assets/cave/cave_hexapod.xml`)
      .then((response) => {
        if (!cancelled) {
          const hasCaveScene = response.ok || response.status === 304
          onLoadDetail?.(
            hasCaveScene
              ? 'Cave scene found; loading cave assets'
              : 'No cave scene found; loading static hexapod',
          )
          if (!hasCaveScene) {
            window.sessionStorage.removeItem(CAVE_DISABLED_SESSION_KEY)
          }
          setSceneFile(hasCaveScene ? CAVE_SCENE_FILE : DEFAULT_SCENE_FILE)
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          onLoadDetail?.(
            isAbortError(error)
              ? 'Cave scene check timed out; loading static hexapod'
              : 'Cave scene check failed; loading static hexapod',
          )
          setSceneFile(DEFAULT_SCENE_FILE)
        }
      })

    return () => {
      cancelled = true
    }
  }, [onLoadDetail])

  useEffect(() => {
    if (!sceneFile) return
    sceneReadyRef.current = false
    onLoadDetail?.(
      sceneFile === CAVE_SCENE_FILE
        ? 'Compiling cave scene in browser'
        : 'Compiling static hexapod scene',
    )
  }, [onLoadDetail, sceneFile])

  useEffect(() => {
    if (sceneFile !== CAVE_SCENE_FILE) return

    const timeout = window.setTimeout(() => {
      if (sceneReadyRef.current) return
      window.sessionStorage.setItem(CAVE_DISABLED_SESSION_KEY, '1')
      onLoadDetail?.('Cave compile timed out; reloading static hexapod')
      window.location.reload()
    }, CAVE_LOAD_TIMEOUT_MS)

    return () => {
      window.clearTimeout(timeout)
    }
  }, [onLoadDetail, sceneFile])

  const config = useMemo(() => {
    if (!sceneFile) return null
    const baseUrl = import.meta.env.VITE_HEXY_BE_URL ?? DEFAULT_BACKEND_URL
    return buildHexyConfig(baseUrl, sceneFile)
  }, [sceneFile])

  const handleMujocoError = (error: unknown, source: string) => {
    if (import.meta.env.DEV) {
      console.error(`[mujoco] ${source} error`, error)
    }

    if (sceneFile === CAVE_SCENE_FILE) {
      sceneReadyRef.current = false
      window.sessionStorage.setItem(CAVE_DISABLED_SESSION_KEY, '1')
      onLoadDetail?.('Cave scene failed; loading static hexapod')
      setSceneFile(DEFAULT_SCENE_FILE)
      return
    }

    onLoadError?.(error instanceof Error ? error.message : String(error))
  }

  if (!config) return null

  return (
    <MujocoProvider
      onError={(error) => handleMujocoError(error, 'provider')}
    >
      <div ref={mainViewRef} className="hexy-scene">
        <CameraTransitionOverlay variant="main" />
        <RobotOverlay variant="main" visible={showMainRobotOverlay} />
      </div>
      <Canvas
        className="hexy-canvas-root"
        dpr={1}
        gl={{ antialias: true, powerPreference: 'high-performance' }}
        onCreated={({ gl }) => {
          gl.setClearColor('#292929')
        }}
      >
        <MujocoPhysics
          key={sceneFile}
          config={config}
          paused
          onError={(error) => handleMujocoError(error, 'scene')}
          onReady={() => {
            sceneReadyRef.current = true
            onLoadDetail?.('MuJoCo scene ready')
            if (import.meta.env.DEV) {
              console.info('[mujoco] scene ready')
            }
            onMainReady?.()
          }}
        >
          <SceneStage
            mainViewRef={mainViewRef}
            pipViewRef={pipViewRef}
            mainCameraPreset={mainCameraPreset}
            pipCameraPreset={pipSlot}
            showPip={renderPip}
            viewMode={viewMode}
          />
        </MujocoPhysics>
      </Canvas>
      {showPip && (
        <PipCameraFrame ref={pipViewRef} collapsed={pipCollapsed} hasCanvas={renderPip}>
          {showPipRobotOverlay && <RobotOverlay variant="pip" />}
        </PipCameraFrame>
      )}
    </MujocoProvider>
  )
}

export default RobotScene
