from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Any, Callable

logger = logging.getLogger("uvicorn.error")

_VALID_ACTIONS = frozenset({"w", "a", "s", "d"})
_STOP_ACTIONS = frozenset({"stop", "none", ""})
_TURN_ACTIONS = frozenset({"a", "d"})
# Batches of 52ms per command: A/D limited to ~4 batches (~208ms) per LLM command
_MAX_TURN_BATCHES = 4


class AutonomousAgent:
    """
    Three-thread autonomous search-and-rescue agent.

    Phase 1 – Physics (spinal cord):
        Dequeues {"action": "w", "n_steps": 20} commands and executes them via
        the MuJoCo IK controller. Holds sim_lock while stepping so manual WASD
        control and rendering don't race.

    Phase 2 – Sensors (autonomic nervous system):
        Polls DINO + cached audio every AGENT_SENSOR_INTERVAL_SEC seconds.
        If something is detected it pushes a context string to the LLM queue.

    Phase 3 – Brain (LLM):
        Blocks on the prompt queue. Runs Qwen inference (~3 s), parses the JSON
        action, and drops it into the action queue for Phase 1 to execute.
    """

    def __init__(self) -> None:
        # action_queue: LLM → physics
        self._action_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=2)
        # prompt_queue: sensors / operator → LLM
        self._prompt_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=8)

        self._running = False
        self._threads: list[threading.Thread] = []

        # Injected at start()
        self._simulator = None
        self._dino_service = None
        self._qwen_service = None
        self._sim_lock: threading.Lock | None = None
        self._build_control_fn: Callable | None = None
        self._render_fn: Callable | None = None

        # Sensor state (written by sensor thread + whisper WebSocket)
        self._state_lock = threading.Lock()
        self._latest_audio: dict[str, Any] = {}
        self._last_audio_ts: float = 0.0
        self._last_detections: list[dict] = []

        # Agent behavioural state
        self._paused: bool = False
        self._last_action: str = ""
        self._last_reasoning: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    def start(
        self,
        simulator,
        dino_service,
        qwen_service,
        sim_lock: threading.Lock,
        build_control_fn: Callable,
        render_fn: Callable | None = None,
    ) -> None:
        if self._running:
            logger.warning("[agent] already running")
            return
        self._simulator = simulator
        self._dino_service = dino_service
        self._qwen_service = qwen_service
        self._sim_lock = sim_lock
        self._build_control_fn = build_control_fn
        self._render_fn = render_fn
        self._running = True

        for target, name in [
            (self._physics_loop, "agent-physics"),
            (self._sensor_loop, "agent-sensor"),
            (self._llm_loop, "agent-llm"),
        ]:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

        logger.info("[agent] autonomous agent started (3 threads)")

    def stop(self) -> None:
        self._running = False
        self._threads.clear()
        logger.info("[agent] stop requested")

    def push_instruction(self, text: str) -> None:
        """Push an operator instruction directly to the LLM queue."""
        try:
            self._prompt_queue.put_nowait({"type": "instruction", "text": text})
            logger.info("[agent] queued instruction: %.80s", text)
        except queue.Full:
            logger.warning("[agent] prompt queue full — dropping instruction")

    def update_audio(self, chunk: dict[str, Any]) -> None:
        """Called by the whisper WebSocket to update the latest transcript."""
        with self._state_lock:
            self._latest_audio = chunk
            self._last_audio_ts = time.time()

    def pause(self) -> None:
        """Stop all autonomous movement and block sensor→LLM pipeline."""
        self._paused = True
        # Drain any queued movement commands
        while True:
            try:
                self._action_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("[agent] paused")

    def resume(self) -> None:
        """Resume autonomous operation."""
        self._paused = False
        logger.info("[agent] resumed")

    def get_state(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "paused": self._paused,
            "last_action": self._last_action,
            "last_reasoning": self._last_reasoning,
        }

    # ------------------------------------------------------------------
    # Phase 1: Physics loop
    # ------------------------------------------------------------------

    def _physics_loop(self) -> None:
        logger.info("[agent-physics] started")
        current_key: str | None = None
        turn_batch_count: int = 0
        _BATCH_SEC = 0.052  # ~quarter gait cycle at 4.8 Hz

        while self._running:
            # Non-blocking check for a new command from the LLM
            try:
                cmd = self._action_queue.get_nowait()
                new_key = str(cmd.get("action", "")).lower().strip()
                if new_key in _VALID_ACTIONS:
                    if new_key != current_key:
                        logger.info("[agent-physics] direction → %s", new_key)
                    current_key = new_key
                    turn_batch_count = 0
                elif new_key in _STOP_ACTIONS:
                    if current_key is not None:
                        logger.info("[agent-physics] stopped")
                    current_key = None
                    turn_batch_count = 0
            except queue.Empty:
                pass

            # Pause overrides everything
            if self._paused:
                current_key = None
                turn_batch_count = 0
                time.sleep(0.05)
                continue

            if current_key is None:
                time.sleep(0.05)
                continue

            # A/D: short burst only — just enough to rotate slightly
            if current_key in _TURN_ACTIONS:
                if turn_batch_count >= _MAX_TURN_BATCHES:
                    current_key = None
                    turn_batch_count = 0
                    time.sleep(0.05)
                    continue
                turn_batch_count += 1
            else:
                turn_batch_count = 0  # W/S: continuous

            try:
                with self._sim_lock:  # type: ignore[union-attr]
                    timestep = float(self._simulator.model.opt.timestep)
                    batch = max(1, round(_BATCH_SEC / timestep))
                    k = current_key
                    self._simulator.drive_key(
                        k,
                        lambda st, _k=k: self._build_control_fn(  # type: ignore[misc]
                            self._simulator, _k, st
                        ),
                        n_steps=batch,
                    )
            except Exception as exc:
                logger.warning("[agent-physics] step failed: %s", exc)

        logger.info("[agent-physics] stopped")

    # ------------------------------------------------------------------
    # Phase 2: Sensor loop
    # ------------------------------------------------------------------

    def _sensor_loop(self) -> None:
        interval = float(os.getenv("AGENT_SENSOR_INTERVAL_SEC", "2.0"))
        audio_stale_sec = float(os.getenv("AGENT_AUDIO_STALE_SEC", "10.0"))
        # Bbox area fraction threshold above which a survivor is considered "near"
        survivor_near_area = float(os.getenv("AGENT_SURVIVOR_NEAR_AREA", "0.20"))
        camera_name = os.getenv("GDINO_CAMERA_DEFAULT")
        logger.info("[agent-sensor] started  interval=%.1fs", interval)

        while self._running:
            time.sleep(interval)
            if not self._running:
                break

            # Skip sensing while paused — don't feed new prompts to the LLM
            if self._paused:
                continue

            context_parts: list[str] = []

            # Vision
            try:
                dino = self._dino_service
                if dino is not None and dino._model is not None:
                    cam_id = self._simulator.resolve_camera(camera_name)[0]
                    if self._render_fn is not None:
                        # Route through the shared render executor so the EGL context
                        # stays in one thread (avoids EGL_BAD_ACCESS with mujoco_detections)
                        frame = self._render_fn(640, 360, cam_id)
                    else:
                        with self._sim_lock:  # type: ignore[union-attr]
                            frame = self._simulator.render_rgb_with_camera_id(640, 360, cam_id)
                    detections = dino.detect(frame, dino.build_config())
                    with self._state_lock:
                        self._last_detections = detections
                    if detections:
                        det_text = "; ".join(
                            f"{d.get('label', '?')} conf={d.get('confidence', 0):.2f}"
                            for d in detections
                        )
                        context_parts.append(f"[VISION] {det_text}")

                        # Auto-stop when survivor bbox fills > survivor_near_area of frame
                        for det in detections:
                            bbox = det.get("bbox") or []
                            if len(bbox) == 4:
                                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                if area > survivor_near_area:
                                    logger.info(
                                        "[agent-sensor] survivor near (bbox area=%.2f) — auto-pausing",
                                        area,
                                    )
                                    self.pause()
                                    try:
                                        self._prompt_queue.put_nowait({
                                            "type": "sensor",
                                            "text": f"[SURVIVOR_NEAR] bbox area={area:.2f} — stopped. Awaiting operator.",
                                        })
                                    except queue.Full:
                                        pass
                                    break
            except Exception as exc:
                logger.debug("[agent-sensor] vision error: %s", exc)

            # Audio
            now = time.time()
            with self._state_lock:
                audio = self._latest_audio.copy()
                audio_ts = self._last_audio_ts

            if audio and (now - audio_ts) < audio_stale_sec:
                transcript = audio.get("transcript", "")
                direction = audio.get("direction", "unknown")
                dist = audio.get("distance_m", "?")
                if transcript:
                    context_parts.append(
                        f'[AUDIO] "{transcript}" from {direction} ~{dist}m'
                    )

            if context_parts:
                text = " | ".join(context_parts)
                try:
                    self._prompt_queue.put_nowait({"type": "sensor", "text": text})
                    logger.info("[agent-sensor] queued: %s", text)
                except queue.Full:
                    pass  # LLM is busy; drop this sensor tick

        logger.info("[agent-sensor] stopped")

    # ------------------------------------------------------------------
    # Phase 3: LLM loop
    # ------------------------------------------------------------------

    def _llm_loop(self) -> None:
        from .agent_prompt import get_agent_prompt
        from .qwen25vl_service import Qwen25VlRequest

        system_prompt = get_agent_prompt()
        logger.info("[agent-llm] started")

        while self._running:
            try:
                item = self._prompt_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            prompt_text = str(item.get("text", "")).strip()
            if not prompt_text:
                continue

            qwen = self._qwen_service
            if qwen is None or not qwen.is_ready:
                logger.debug("[agent-llm] model not ready — dropping prompt")
                continue

            with self._state_lock:
                detections = list(self._last_detections)
                audio = self._latest_audio.copy()

            user_payload = {
                "instruction": prompt_text,
                "detections": detections,
                "audio": {
                    "transcript": audio.get("transcript"),
                    "direction": audio.get("direction"),
                    "distance_m": audio.get("distance_m"),
                },
            }

            try:
                logger.info("[agent-llm] inferring  type=%s", item.get("type"))
                result = qwen.infer(
                    Qwen25VlRequest(
                        system_prompt=system_prompt,
                        user_prompt=json.dumps(user_payload, ensure_ascii=True),
                        max_new_tokens=200,
                        temperature=0.2,
                        top_p=0.9,
                    )
                )
                agent_json = result.get("json") or {}
                action = str(agent_json.get("action", "")).lower().strip()
                reasoning = str(agent_json.get("reasoning", ""))
                logger.info(
                    "[agent-llm] → action=%s  reason=%.80s",
                    action, reasoning,
                )
                # Persist for /agent/state polling
                self._last_action = action
                self._last_reasoning = reasoning

                if action in _VALID_ACTIONS or action in _STOP_ACTIONS:
                    cmd = {"action": action}
                    try:
                        self._action_queue.put_nowait(cmd)
                    except queue.Full:
                        # Replace stale queued action with fresher decision
                        try:
                            self._action_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self._action_queue.put_nowait(cmd)

            except Exception as exc:
                logger.warning("[agent-llm] inference failed: %s", exc)

        logger.info("[agent-llm] stopped")
