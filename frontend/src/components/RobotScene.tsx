import { Canvas } from '@react-three/fiber'
import { MujocoPhysics, MujocoProvider } from 'mujoco-react'
import type { SceneConfig } from 'mujoco-react'
import { useMemo, useRef } from 'react'
import { useViewportStore } from '@/stores/useViewportStore'
import { SceneStage } from './SceneStage'
import { PipCameraFrame } from './PipCameraFrame'
import { CameraTransitionOverlay } from './CameraTransitionOverlay'

const DEFAULT_BACKEND_URL = ''
const DEFAULT_SCENE_FILE = 'rl/models/hexapod_static.xml'

function buildHexyConfig(baseUrl: string): SceneConfig {
  const normalizedBase = baseUrl.endsWith('/')
    ? baseUrl.slice(0, -1)
    : baseUrl
  const sceneFile = import.meta.env.VITE_HEXY_SCENE_FILE ?? DEFAULT_SCENE_FILE

  return {
    src: `${normalizedBase}/assets/`,
    sceneFile,
  }
}

type RobotSceneProps = {
  onMainReady?: () => void
  onLoadError?: (message: string) => void
  showPip?: boolean
}

export function RobotScene({
  onMainReady,
  onLoadError,
  showPip = true,
}: RobotSceneProps) {
  const pipSlot = useViewportStore((s) => s.pipSlot)
  const pipCollapsed = useViewportStore((s) => s.pipCollapsed)
  const mainViewRef = useRef<HTMLDivElement | null>(null)
  const pipViewRef = useRef<HTMLDivElement | null>(null)
  const mainCameraPreset = pipSlot === 'robotPOV' ? 'orbit' : 'robotPOV'
  const renderPip = showPip && !pipCollapsed
  const config = useMemo(() => {
    const baseUrl = import.meta.env.VITE_HEXY_BE_URL ?? DEFAULT_BACKEND_URL
    return buildHexyConfig(baseUrl)
  }, [])

  return (
    <MujocoProvider
      onError={(error) => {
        if (import.meta.env.DEV) {
          console.error('[mujoco] provider error', error)
        }
        onLoadError?.(error.message)
      }}
    >
      <div ref={mainViewRef} className="hexy-scene">
        <CameraTransitionOverlay variant="main" />
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
          config={config}
          paused
          onError={(error) => {
            if (import.meta.env.DEV) {
              console.error('[mujoco] scene error', error)
            }
            onLoadError?.(error.message)
          }}
          onReady={() => {
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
          />
        </MujocoPhysics>
      </Canvas>
      {showPip && (
        <PipCameraFrame ref={pipViewRef} collapsed={pipCollapsed} hasCanvas={renderPip} />
      )}
    </MujocoProvider>
  )
}

export default RobotScene
