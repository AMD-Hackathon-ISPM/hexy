import { useEffect, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { findBodyByName, useMujoco } from 'mujoco-react'
import { useRobotStatusStore, type SurvivorAudioEvent } from '@/stores/useRobotStatusStore'
import * as THREE from 'three'

const SURVIVOR_AUDIO_FILES: Record<string, string> = {
  help: '/audio/survivors/help.mp3',
  'save me': '/audio/survivors/save-me.mp3',
  'over here': '/audio/survivors/over-here.mp3',
}

const SURVIVOR_AUDIO_Z_OFFSET = 0.75
const REVERB_DURATION_SEC = 2.4
const REVERB_DECAY = 2.7
const REVERB_WET_GAIN = 0.18
const DRY_GAIN = 0.9
const LISTENER_UPDATE_INTERVAL_SEC = 1 / 30

type SurvivorSpatialAudioProps = {
  listenerCamera: THREE.PerspectiveCamera
}

type AudioRuntime = {
  context: AudioContext
  convolver: ConvolverNode
  buffers: Map<string, AudioBuffer>
  missing: Set<string>
  unlocked: boolean
}

type ModernAudioListener = AudioListener & {
  positionX?: AudioParam
  positionY?: AudioParam
  positionZ?: AudioParam
  forwardX?: AudioParam
  forwardY?: AudioParam
  forwardZ?: AudioParam
  upX?: AudioParam
  upY?: AudioParam
  upZ?: AudioParam
}

type ModernPannerNode = PannerNode & {
  positionX?: AudioParam
  positionY?: AudioParam
  positionZ?: AudioParam
}

function normalizePhrase(phrase: string) {
  return phrase.trim().toLowerCase().replace(/\s+/g, ' ')
}

function createCaveImpulse(context: AudioContext) {
  const length = Math.max(1, Math.floor(context.sampleRate * REVERB_DURATION_SEC))
  const impulse = context.createBuffer(2, length, context.sampleRate)

  for (let channel = 0; channel < impulse.numberOfChannels; channel += 1) {
    const data = impulse.getChannelData(channel)
    for (let i = 0; i < length; i += 1) {
      const progress = i / length
      const earlyReflection = i < context.sampleRate * 0.08 ? 1.35 : 1.0
      data[i] = (Math.random() * 2 - 1) * ((1 - progress) ** REVERB_DECAY) * earlyReflection
    }
  }

  return impulse
}

function setParam(param: AudioParam | undefined, value: number, context: AudioContext) {
  param?.setValueAtTime(value, context.currentTime)
}

function setPannerPosition(
  panner: PannerNode,
  position: THREE.Vector3,
  context: AudioContext,
) {
  const modernPanner = panner as ModernPannerNode
  if (modernPanner.positionX && modernPanner.positionY && modernPanner.positionZ) {
    setParam(modernPanner.positionX, position.x, context)
    setParam(modernPanner.positionY, position.y, context)
    setParam(modernPanner.positionZ, position.z, context)
    return
  }

  panner.setPosition(position.x, position.y, position.z)
}

function setListenerPose(
  context: AudioContext,
  position: THREE.Vector3,
  forward: THREE.Vector3,
  up: THREE.Vector3,
) {
  const listener = context.listener as ModernAudioListener
  if (
    listener.positionX &&
    listener.positionY &&
    listener.positionZ &&
    listener.forwardX &&
    listener.forwardY &&
    listener.forwardZ &&
    listener.upX &&
    listener.upY &&
    listener.upZ
  ) {
    setParam(listener.positionX, position.x, context)
    setParam(listener.positionY, position.y, context)
    setParam(listener.positionZ, position.z, context)
    setParam(listener.forwardX, forward.x, context)
    setParam(listener.forwardY, forward.y, context)
    setParam(listener.forwardZ, forward.z, context)
    setParam(listener.upX, up.x, context)
    setParam(listener.upY, up.y, context)
    setParam(listener.upZ, up.z, context)
    return
  }

  listener.setPosition(position.x, position.y, position.z)
  listener.setOrientation(forward.x, forward.y, forward.z, up.x, up.y, up.z)
}

function disconnectNode(node: AudioNode) {
  try {
    node.disconnect()
  } catch {
    // The node may already be disconnected after context shutdown.
  }
}

function yieldToInput() {
  return new Promise<void>((resolve) => {
    window.setTimeout(resolve, 0)
  })
}

export function SurvivorSpatialAudio({ listenerCamera }: SurvivorSpatialAudioProps) {
  const mujoco = useMujoco()
  const survivorAudioEvent = useRobotStatusStore((s) => s.survivorAudioEvent)
  const audioRuntimeRef = useRef<AudioRuntime | null>(null)
  const bodyIdsRef = useRef(new Map<string, number>())
  const cameraPosition = useRef(new THREE.Vector3())
  const cameraForward = useRef(new THREE.Vector3())
  const cameraUp = useRef(new THREE.Vector3())
  const sourcePosition = useRef(new THREE.Vector3())
  const listenerUpdateElapsedRef = useRef(0)

  const getAudioRuntime = () => {
    if (audioRuntimeRef.current) return audioRuntimeRef.current

    const AudioContextConstructor =
      window.AudioContext ??
      (window as typeof window & { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext
    if (!AudioContextConstructor) return null

    const context = new AudioContextConstructor()
    const convolver = context.createConvolver()
    const wetGain = context.createGain()
    wetGain.gain.value = REVERB_WET_GAIN
    convolver.buffer = createCaveImpulse(context)
    convolver.connect(wetGain)
    wetGain.connect(context.destination)

    audioRuntimeRef.current = {
      context,
      convolver,
      buffers: new Map(),
      missing: new Set(),
      unlocked: context.state === 'running',
    }
    return audioRuntimeRef.current
  }

  const preloadAudioFiles = async (runtime: AudioRuntime) => {
    for (const phrase of Object.keys(SURVIVOR_AUDIO_FILES)) {
      await loadAudioBuffer(runtime, phrase)
      await yieldToInput()
    }
  }

  const loadAudioBuffer = async (runtime: AudioRuntime, phrase: string) => {
    const key = normalizePhrase(phrase)
    const existing = runtime.buffers.get(key)
    if (existing) return existing
    if (runtime.missing.has(key)) return null

    const url = SURVIVOR_AUDIO_FILES[key]
    if (!url) {
      runtime.missing.add(key)
      return null
    }

    try {
      const response = await fetch(url)
      if (!response.ok) {
        runtime.missing.add(key)
        return null
      }

      const buffer = await response.arrayBuffer()
      const audioBuffer = await runtime.context.decodeAudioData(buffer)
      runtime.buffers.set(key, audioBuffer)
      return audioBuffer
    } catch {
      runtime.missing.add(key)
      return null
    }
  }

  const resolveBodyId = (sourceId: string) => {
    const cached = bodyIdsRef.current.get(sourceId)
    if (cached !== undefined) return cached

    const model = mujoco.isReady ? mujoco.mjModelRef.current : null
    if (!model) return null

    const bodyId = findBodyByName(model, sourceId)
    if (bodyId < 0) return null

    bodyIdsRef.current.set(sourceId, bodyId)
    return bodyId
  }

  const playSurvivorAudioEvent = async (event: SurvivorAudioEvent) => {
    const runtime = audioRuntimeRef.current
    if (!runtime?.unlocked || runtime.context.state !== 'running') return

    const data = mujoco.isReady ? mujoco.mjDataRef.current : null
    if (!data) return

    const bodyId = resolveBodyId(event.sourceId)
    if (bodyId === null) return

    const audioBuffer = await loadAudioBuffer(runtime, event.phrase)
    if (!audioBuffer) return

    const i3 = bodyId * 3
    sourcePosition.current.set(
      data.xpos[i3],
      data.xpos[i3 + 1],
      data.xpos[i3 + 2] + SURVIVOR_AUDIO_Z_OFFSET,
    )

    const context = runtime.context
    const source = context.createBufferSource()
    const panner = context.createPanner()
    const dryGain = context.createGain()

    source.buffer = audioBuffer
    panner.panningModel = 'HRTF'
    panner.distanceModel = 'exponential'
    panner.refDistance = 2
    panner.maxDistance = 45
    panner.rolloffFactor = 1.35
    panner.coneInnerAngle = 360
    panner.coneOuterAngle = 360
    dryGain.gain.value = DRY_GAIN

    setPannerPosition(panner, sourcePosition.current, context)

    source.connect(panner)
    panner.connect(dryGain)
    dryGain.connect(context.destination)
    panner.connect(runtime.convolver)

    source.onended = () => {
      disconnectNode(source)
      disconnectNode(panner)
      disconnectNode(dryGain)
    }
    source.start()
  }

  useEffect(() => {
    const unlockAudio = () => {
      window.removeEventListener('pointerdown', unlockAudio)
      window.removeEventListener('keydown', unlockAudio)

      const runtime = getAudioRuntime()
      if (!runtime) return

      void runtime.context.resume().then(() => {
        runtime.unlocked = runtime.context.state === 'running'
        if (runtime.unlocked) {
          void preloadAudioFiles(runtime)
        }
      })
    }

    window.addEventListener('pointerdown', unlockAudio)
    window.addEventListener('keydown', unlockAudio)

    return () => {
      window.removeEventListener('pointerdown', unlockAudio)
      window.removeEventListener('keydown', unlockAudio)
    }
  }, [])

  useEffect(() => () => {
    const runtime = audioRuntimeRef.current
    audioRuntimeRef.current = null
    if (runtime) {
      void runtime.context.close()
    }
  }, [])

  useEffect(() => {
    if (!survivorAudioEvent) return
    void playSurvivorAudioEvent(survivorAudioEvent)
  }, [survivorAudioEvent])

  useFrame((_, delta) => {
    const runtime = audioRuntimeRef.current
    if (!runtime?.unlocked || runtime.context.state !== 'running') return
    listenerUpdateElapsedRef.current += delta
    if (listenerUpdateElapsedRef.current < LISTENER_UPDATE_INTERVAL_SEC) return
    listenerUpdateElapsedRef.current = 0

    listenerCamera.getWorldPosition(cameraPosition.current)
    listenerCamera.getWorldDirection(cameraForward.current)
    cameraUp.current.copy(listenerCamera.up).applyQuaternion(listenerCamera.quaternion).normalize()
    setListenerPose(
      runtime.context,
      cameraPosition.current,
      cameraForward.current,
      cameraUp.current,
    )
  }, 2)

  return null
}

export default SurvivorSpatialAudio
