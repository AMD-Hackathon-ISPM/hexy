import { useEffect, useMemo, useRef, type RefObject } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { findBodyByName, useMujoco } from 'mujoco-react'
import type { MujocoState } from '@/lib/backendClient'
import { useViewportStore, type CameraPreset, type ViewMode } from '@/stores/useViewportStore'
import { useRobotStatusStore } from '@/stores/useRobotStatusStore'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'

const cameraConfig = {
  orbit: {
    position: [2.8, -2.2, 1.65] as [number, number, number],
    up: [0, 0, 1] as [number, number, number],
    fov: 42,
  },
  robotPOV: {
    position: [0.0, 0.0, 0.65] as [number, number, number],
    up: [0, 0, 1] as [number, number, number],
    fov: 45,
  },
} as const

const ORBIT_TARGET_FALLBACK = [0, 0, 0.22] as [number, number, number]
const HEXAPOD_ORBIT_TARGET_OFFSET = [0, 0, 0.155] as [number, number, number]
const ORBIT_CAMERA_OFFSET = [2.8, -2.2, 1.43] as [number, number, number]
const MUJOCO_STATE_INTERPOLATION_MS = 70

const cameraClip = {
  orbit: {
    near: 0.1,
    far: 1000,
  },
  robotPOV: {
    near: 0.01,
    far: 30,
  },
} as const

type SceneStageProps = {
  mainViewRef: RefObject<HTMLDivElement | null>
  pipViewRef: RefObject<HTMLDivElement | null>
  mainCameraPreset: CameraPreset
  pipCameraPreset: CameraPreset
  showPip: boolean
  viewMode: ViewMode
}

type ViewportBounds = {
  left: number
  bottom: number
  width: number
  height: number
  aspect: number
}

type ViewportRect = Pick<DOMRectReadOnly, 'left' | 'right' | 'top' | 'bottom'>

function readViewportRect(rect: DOMRectReadOnly): ViewportRect {
  return {
    left: rect.left,
    right: rect.right,
    top: rect.top,
    bottom: rect.bottom,
  }
}

function getViewportBounds(
  rect: ViewportRect,
  canvasRect: ViewportRect,
): ViewportBounds | null {
  const clippedLeft = Math.max(rect.left, canvasRect.left)
  const clippedRight = Math.min(rect.right, canvasRect.right)
  const clippedTop = Math.max(rect.top, canvasRect.top)
  const clippedBottom = Math.min(rect.bottom, canvasRect.bottom)
  const cssWidth = clippedRight - clippedLeft
  const cssHeight = clippedBottom - clippedTop

  if (cssWidth <= 0 || cssHeight <= 0) return null

  return {
    left: Math.round(clippedLeft - canvasRect.left),
    bottom: Math.round(canvasRect.bottom - clippedBottom),
    width: Math.round(cssWidth),
    height: Math.round(cssHeight),
    aspect: cssWidth / cssHeight,
  }
}

function resetSceneCamera(
  camera: THREE.PerspectiveCamera,
  cameraPreset: CameraPreset,
  orbitTarget?: THREE.Vector3,
) {
  camera.position.set(...cameraConfig[cameraPreset].position)
  camera.up.set(...cameraConfig[cameraPreset].up)
  camera.fov = cameraConfig[cameraPreset].fov
  camera.near = cameraClip[cameraPreset].near
  camera.far = cameraClip[cameraPreset].far

  if (cameraPreset === 'orbit') {
    const target = orbitTarget ?? new THREE.Vector3(...ORBIT_TARGET_FALLBACK)
    camera.position.set(
      target.x + ORBIT_CAMERA_OFFSET[0],
      target.y + ORBIT_CAMERA_OFFSET[1],
      target.z + ORBIT_CAMERA_OFFSET[2],
    )
    camera.lookAt(target)
  }

  camera.updateProjectionMatrix()
  camera.updateMatrixWorld()
}

function createSceneCamera(cameraPreset: CameraPreset) {
  const camera = new THREE.PerspectiveCamera(
    cameraConfig[cameraPreset].fov,
    16 / 9,
    cameraClip[cameraPreset].near,
    cameraClip[cameraPreset].far,
  )

  resetSceneCamera(camera, cameraPreset)

  return camera
}

function renderCameraViewport(
  gl: THREE.WebGLRenderer,
  scene: THREE.Scene,
  rect: ViewportRect,
  canvasRect: ViewportRect,
  camera: THREE.PerspectiveCamera,
) {
  const viewport = getViewportBounds(rect, canvasRect)
  if (!viewport) return

  camera.aspect = viewport.aspect
  camera.updateProjectionMatrix()
  camera.updateMatrixWorld()

  gl.setViewport(viewport.left, viewport.bottom, viewport.width, viewport.height)
  gl.setScissor(viewport.left, viewport.bottom, viewport.width, viewport.height)
  gl.setScissorTest(true)
  gl.clear(true, true, true)
  gl.render(scene, camera)
}

type ReflectorMesh = THREE.Mesh & {
  isReflector?: boolean
  getRenderTarget?: () => THREE.WebGLRenderTarget
}

function disableSceneShadowsAndReflections(scene: THREE.Scene) {
  let hasRenderableMesh = false

  scene.traverse((object) => {
    object.castShadow = false
    object.receiveShadow = false

    const mesh = object as ReflectorMesh
    if (mesh.isMesh) hasRenderableMesh = true
    if (!mesh.isReflector || mesh.userData.hexyEffectsDisabled) return

    const material = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material
    const color =
      material && 'color' in material && material.color instanceof THREE.Color
        ? material.color.clone()
        : new THREE.Color(0x999999)

    mesh.onBeforeRender = () => {}
    mesh.material = new THREE.MeshStandardMaterial({
      color,
      roughness: 1,
      metalness: 0,
    })
    mesh.getRenderTarget?.().dispose()
    material?.dispose()
    mesh.userData.hexyEffectsDisabled = true
  })

  return hasRenderableMesh
}

function MujocoStateSync() {
  const { api, isReady, mjModelRef } = useMujoco()
  const lastUpdateRef = useRef<number | null>(null)
  const mismatchWarnedRef = useRef(false)
  const fromQposRef = useRef<Float64Array | null>(null)
  const toQposRef = useRef<Float64Array | null>(null)
  const fromQvelRef = useRef<Float64Array | null>(null)
  const toQvelRef = useRef<Float64Array | null>(null)
  const renderedQposRef = useRef<Float64Array | null>(null)
  const renderedQvelRef = useRef<Float64Array | null>(null)
  const transitionStartedAtRef = useRef(0)
  const transitionSettledRef = useRef(true)

  const applyInterpolatedState = (alpha: number) => {
    if (!api) return

    const fromQpos = fromQposRef.current
    const toQpos = toQposRef.current
    const fromQvel = fromQvelRef.current
    const toQvel = toQvelRef.current
    if (!fromQpos || !toQpos || !fromQvel || !toQvel) return

    const qpos = renderedQposRef.current ?? new Float64Array(toQpos.length)
    const qvel = renderedQvelRef.current ?? new Float64Array(toQvel.length)
    renderedQposRef.current = qpos
    renderedQvelRef.current = qvel

    for (let i = 0; i < toQpos.length; i += 1) {
      qpos[i] = fromQpos[i] + (toQpos[i] - fromQpos[i]) * alpha
    }
    if (qpos.length >= 7) {
      const rootQuatLength = Math.hypot(qpos[3], qpos[4], qpos[5], qpos[6])
      if (rootQuatLength > 0) {
        qpos[3] /= rootQuatLength
        qpos[4] /= rootQuatLength
        qpos[5] /= rootQuatLength
        qpos[6] /= rootQuatLength
      }
    }
    for (let i = 0; i < toQvel.length; i += 1) {
      qvel[i] = fromQvel[i] + (toQvel[i] - fromQvel[i]) * alpha
    }

    api.setQpos(qpos)
    api.setQvel(qvel)
  }

  useEffect(() => {
    if (!isReady) return
    api.setPaused(true)
  }, [api, isReady])

  useEffect(() => {
    if (!isReady) return

    const applyMujocoState = (mujocoState?: MujocoState) => {
      if (!mujocoState) return
      if (lastUpdateRef.current === mujocoState.updated_at) return

      const model = mjModelRef.current
      if (model && mujocoState.qpos.length !== model.nq) {
        if (!mismatchWarnedRef.current && import.meta.env.DEV) {
          console.warn('[mujoco] ignored streamed qpos with mismatched model size', {
            streamed: mujocoState.qpos.length,
            browserModel: model.nq,
          })
          mismatchWarnedRef.current = true
        }
        return
      }
      if (model && mujocoState.qvel.length !== model.nv) {
        if (!mismatchWarnedRef.current && import.meta.env.DEV) {
          console.warn('[mujoco] ignored streamed qvel with mismatched model size', {
            streamed: mujocoState.qvel.length,
            browserModel: model.nv,
          })
          mismatchWarnedRef.current = true
        }
        return
      }

      const nextQpos = Float64Array.from(mujocoState.qpos)
      const nextQvel = Float64Array.from(mujocoState.qvel)
      const renderedQpos = renderedQposRef.current
      const renderedQvel = renderedQvelRef.current

      fromQposRef.current = renderedQpos
        ? new Float64Array(renderedQpos)
        : new Float64Array(nextQpos)
      fromQvelRef.current = renderedQvel
        ? new Float64Array(renderedQvel)
        : new Float64Array(nextQvel)
      toQposRef.current = nextQpos
      toQvelRef.current = nextQvel
      transitionStartedAtRef.current = performance.now()
      transitionSettledRef.current = false

      if (!renderedQpos || !renderedQvel) {
        applyInterpolatedState(1)
        transitionSettledRef.current = true
      }

      lastUpdateRef.current = mujocoState.updated_at
    }

    applyMujocoState(useRobotStatusStore.getState().mujocoState)

    return useRobotStatusStore.subscribe((state, previousState) => {
      if (state.mujocoState === previousState.mujocoState) return
      applyMujocoState(state.mujocoState)
    })
  }, [api, isReady, mjModelRef])

  useFrame(() => {
    if (!isReady || !api || transitionSettledRef.current) return

    const elapsed = performance.now() - transitionStartedAtRef.current
    const alpha = Math.min(1, elapsed / MUJOCO_STATE_INTERPOLATION_MS)
    applyInterpolatedState(alpha)

    if (alpha >= 1) {
      transitionSettledRef.current = true
    }
  })

  return null
}

type RobotPovCameraProps = {
  camera: THREE.PerspectiveCamera
  offset: THREE.Vector3
  forward: THREE.Vector3
}

function RobotPovCamera({ camera, offset, forward }: RobotPovCameraProps) {
  const mujoco = useMujoco()
  const bodyIdRef = useRef(-1)
  const bodyPos = useRef(new THREE.Vector3())
  const bodyQuat = useRef(new THREE.Quaternion())
  const tmpPos = useRef(new THREE.Vector3())
  const tmpLook = useRef(new THREE.Vector3())
  const forwardRef = useRef(new THREE.Vector3(0.6, 0.0, 0.12))

  useEffect(() => {
    if (!mujoco.isReady) return
    const model = mujoco.mjModelRef.current
    if (!model) return
    bodyIdRef.current = findBodyByName(model, 'hexapod')
  }, [mujoco])

  useEffect(() => {
    forwardRef.current.copy(forward)
  }, [forward])

  useFrame(() => {
    const data = mujoco.isReady ? mujoco.mjDataRef.current : null
    const bodyId = bodyIdRef.current
    if (!data || bodyId < 0) return

    const i3 = bodyId * 3
    const i4 = bodyId * 4

    bodyPos.current.set(data.xpos[i3], data.xpos[i3 + 1], data.xpos[i3 + 2])
    bodyQuat.current.set(
      data.xquat[i4 + 1],
      data.xquat[i4 + 2],
      data.xquat[i4 + 3],
      data.xquat[i4],
    )

    tmpPos.current.copy(offset).applyQuaternion(bodyQuat.current).add(bodyPos.current)
    tmpLook.current.copy(forwardRef.current).applyQuaternion(bodyQuat.current).add(tmpPos.current)

    camera.position.copy(tmpPos.current)
    camera.up.set(0, 0, 1)
    camera.lookAt(tmpLook.current)
    camera.updateMatrixWorld()
  })

  return null
}

function SceneEffectsDisabled() {
  const scene = useThree((state) => state.scene)
  const effectsDisabledRef = useRef(false)

  useFrame(() => {
    if (effectsDisabledRef.current) return
    effectsDisabledRef.current = disableSceneShadowsAndReflections(scene)
  })

  return null
}

function useViewportRect(element: Element | null) {
  const rectRef = useRef<ViewportRect | null>(null)

  useEffect(() => {
    if (!element) return

    const update = () => {
      rectRef.current = readViewportRect(element.getBoundingClientRect())
    }

    update()

    const observer = new ResizeObserver(update)
    observer.observe(element)
    window.addEventListener('resize', update)
    window.addEventListener('scroll', update, true)

    return () => {
      observer.disconnect()
      window.removeEventListener('resize', update)
      window.removeEventListener('scroll', update, true)
    }
  }, [element])

  return rectRef
}

function useRefViewportRect<T extends Element>(ref: RefObject<T | null>) {
  const rectRef = useRef<ViewportRect | null>(null)

  useEffect(() => {
    const element = ref.current
    if (!element) return

    const update = () => {
      rectRef.current = readViewportRect(element.getBoundingClientRect())
    }

    update()

    const observer = new ResizeObserver(update)
    observer.observe(element)
    window.addEventListener('resize', update)
    window.addEventListener('scroll', update, true)

    return () => {
      observer.disconnect()
      window.removeEventListener('resize', update)
      window.removeEventListener('scroll', update, true)
    }
  }, [ref])

  return rectRef
}

function OrbitCameraControls({
  camera,
  enabled,
  freecam,
  targetRef,
  freecamResetNonce,
}: {
  camera: THREE.PerspectiveCamera
  enabled: boolean
  freecam: boolean
  targetRef: RefObject<THREE.Vector3>
  freecamResetNonce: number
}) {
  const gl = useThree((state) => state.gl)
  const mujoco = useMujoco()
  const controlsRef = useRef<OrbitControlsImpl | null>(null)
  const bodyIdRef = useRef(-1)
  const nextTarget = useRef(new THREE.Vector3())
  const targetDelta = useRef(new THREE.Vector3())
  const orbitFocusInitializedRef = useRef(false)
  const freecamFocusInitializedRef = useRef(false)
  const lastFreecamResetNonceRef = useRef(freecamResetNonce)
  const freecamPoseRef = useRef({
    initialized: false,
    position: new THREE.Vector3(),
    target: new THREE.Vector3(),
  })

  const focusCameraOnTarget = (target: THREE.Vector3) => {
    camera.position.set(
      target.x + ORBIT_CAMERA_OFFSET[0],
      target.y + ORBIT_CAMERA_OFFSET[1],
      target.z + ORBIT_CAMERA_OFFSET[2],
    )
    camera.lookAt(target)
    camera.updateMatrixWorld()
    controlsRef.current?.target.copy(target)
    controlsRef.current?.update()
  }

  const saveFreecamPose = () => {
    const controls = controlsRef.current
    if (!controls) return
    freecamPoseRef.current.position.copy(camera.position)
    freecamPoseRef.current.target.copy(controls.target)
    freecamPoseRef.current.initialized = true
  }

  useEffect(() => {
    if (!mujoco.isReady) return
    const model = mujoco.mjModelRef.current
    if (!model) return
    bodyIdRef.current = findBodyByName(model, 'hexapod')
  }, [mujoco.isReady, mujoco.mjModelRef])

  useEffect(() => {
    orbitFocusInitializedRef.current = false

    if (freecam && freecamPoseRef.current.initialized) {
      camera.position.copy(freecamPoseRef.current.position)
      controlsRef.current?.target.copy(freecamPoseRef.current.target)
      controlsRef.current?.update()
      camera.updateMatrixWorld()
      freecamFocusInitializedRef.current = true
      return
    }

    freecamFocusInitializedRef.current = !freecam
    controlsRef.current?.target.copy(targetRef.current)
  }, [camera, freecam, targetRef])

  useFrame(() => {
    const data = mujoco.isReady ? mujoco.mjDataRef.current : null
    const bodyId = bodyIdRef.current
    const hasRobotPose = Boolean(data && bodyId >= 0)

    if (hasRobotPose && data) {
      const i3 = bodyId * 3
      nextTarget.current.set(
        data.xpos[i3] + HEXAPOD_ORBIT_TARGET_OFFSET[0],
        data.xpos[i3 + 1] + HEXAPOD_ORBIT_TARGET_OFFSET[1],
        data.xpos[i3 + 2] + HEXAPOD_ORBIT_TARGET_OFFSET[2],
      )

      targetDelta.current.copy(nextTarget.current).sub(targetRef.current)
      if (targetDelta.current.lengthSq() > 0.00000001) {
        targetRef.current.copy(nextTarget.current)
        if (
          (!freecam && !orbitFocusInitializedRef.current) ||
          (freecam && !freecamFocusInitializedRef.current)
        ) {
          camera.position.add(targetDelta.current)
        }
      }
    }

    if (freecamResetNonce !== lastFreecamResetNonceRef.current) {
      freecamPoseRef.current.initialized = false
      freecamFocusInitializedRef.current = false

      if (!freecam || hasRobotPose) {
        lastFreecamResetNonceRef.current = freecamResetNonce
      }

      if (freecam && hasRobotPose) {
        focusCameraOnTarget(targetRef.current)
        freecamFocusInitializedRef.current = true
        saveFreecamPose()
      }
    }

    if (!freecam || !freecamFocusInitializedRef.current) {
      controlsRef.current?.target.copy(targetRef.current)
      if (!freecam || hasRobotPose) {
        if (!freecam) {
          orbitFocusInitializedRef.current = true
        }
        freecamFocusInitializedRef.current = true
      }
    }
  }, -2)

  useFrame(() => {
    if (freecam && freecamFocusInitializedRef.current) {
      saveFreecamPose()
    }
  })

  return (
    <OrbitControls
      ref={controlsRef}
      key={freecam ? 'freecam' : 'orbital'}
      camera={camera}
      domElement={gl.domElement}
      enableDamping
      enablePan={freecam}
      screenSpacePanning={freecam}
      panSpeed={1.4}
      maxDistance={freecam ? 250 : 100}
      enabled={enabled}
    />
  )
}

type DualViewportRendererProps = {
  mainViewRef: RefObject<HTMLDivElement | null>
  pipViewRef: RefObject<HTMLDivElement | null>
  mainCameraPreset: CameraPreset
  pipCameraPreset: CameraPreset
  orbitCamera: THREE.PerspectiveCamera
  robotCamera: THREE.PerspectiveCamera
  showPip: boolean
}

function DualViewportRenderer({
  mainViewRef,
  pipViewRef,
  mainCameraPreset,
  pipCameraPreset,
  orbitCamera,
  robotCamera,
  showPip,
}: DualViewportRendererProps) {
  const gl = useThree((state) => state.gl)
  const scene = useThree((state) => state.scene)
  const canvasRectRef = useViewportRect(gl.domElement)
  const mainRectRef = useRefViewportRect(mainViewRef)
  const pipRectRef = useRefViewportRect(pipViewRef)

  useFrame(() => {
    const mainElement = mainViewRef.current
    if (!mainElement) return

    const canvasRect =
      canvasRectRef.current ?? readViewportRect(gl.domElement.getBoundingClientRect())
    const mainRect =
      mainRectRef.current ?? readViewportRect(mainElement.getBoundingClientRect())

    gl.setScissorTest(false)
    gl.clear(true, true, true)

    renderCameraViewport(
      gl,
      scene,
      mainRect,
      canvasRect,
      mainCameraPreset === 'orbit' ? orbitCamera : robotCamera,
    )

    const pipElement = pipViewRef.current
    if (showPip && pipElement) {
      const pipRect =
        pipRectRef.current ?? readViewportRect(pipElement.getBoundingClientRect())

      renderCameraViewport(
        gl,
        scene,
        pipRect,
        canvasRect,
        pipCameraPreset === 'orbit' ? orbitCamera : robotCamera,
      )
    }

    gl.setScissorTest(false)
  }, 1)

  return null
}

export function SceneStage({
  mainViewRef,
  pipViewRef,
  mainCameraPreset,
  pipCameraPreset,
  showPip,
  viewMode,
}: SceneStageProps) {
  const orbitCamera = useMemo(() => createSceneCamera('orbit'), [])
  const robotCamera = useMemo(() => createSceneCamera('robotPOV'), [])
  const orbitTargetRef = useRef(new THREE.Vector3(...ORBIT_TARGET_FALLBACK))
  const freecamResetNonce = useViewportStore((s) => s.freecamResetNonce)
  const robotPovOffset = useMemo(() => new THREE.Vector3(0.0, 0.0, 0.65), [])
  const robotPovForward = useMemo(() => new THREE.Vector3(0.0, 1.0, 0.0), [])

  useEffect(() => {
    if (viewMode === 'freecam') return
    if (mainCameraPreset === 'orbit' || pipCameraPreset === 'orbit') {
      resetSceneCamera(orbitCamera, 'orbit', orbitTargetRef.current)
    }
  }, [mainCameraPreset, orbitCamera, pipCameraPreset, viewMode])

  return (
    <>
      <ambientLight intensity={0.7} />
      <directionalLight
        position={[1, 2, 5]}
        intensity={1.2}
      />
      <OrbitCameraControls
        camera={orbitCamera}
        enabled={mainCameraPreset === 'orbit'}
        freecam={viewMode === 'freecam'}
        targetRef={orbitTargetRef}
        freecamResetNonce={freecamResetNonce}
      />
      <RobotPovCamera
        camera={robotCamera}
        offset={robotPovOffset}
        forward={robotPovForward}
      />
      <MujocoStateSync />
      <SceneEffectsDisabled />
      <DualViewportRenderer
        mainViewRef={mainViewRef}
        pipViewRef={pipViewRef}
        mainCameraPreset={mainCameraPreset}
        pipCameraPreset={pipCameraPreset}
        orbitCamera={orbitCamera}
        robotCamera={robotCamera}
        showPip={showPip}
      />
    </>
  )
}

export default SceneStage
