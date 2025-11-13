import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepResult:
    """
    Container for data produced by a single control step of the simulation.
    """

    observation: OrderedDict[str, np.ndarray]
    reward: float
    done: bool
    info: Dict[str, Any]
    action: np.ndarray
    timestamp: float


class ObservationStream:
    """
    Thread-safe buffer that stores the most recent observation (StepResult) along with
    a configurable history for consumers that prefer blocking reads.
    """

    def __init__(self, history: int = 1):
        if history <= 0:
            raise ValueError("ObservationStream history must be a positive integer.")
        self._history = history
        self._buffer = deque(maxlen=history)
        self._condition = threading.Condition()
        self._version = 0

    def publish(self, step: StepResult):
        with self._condition:
            self._buffer.append(step)
            self._version += 1
            self._condition.notify_all()

    def latest(self) -> Optional[StepResult]:
        with self._condition:
            return self._buffer[-1] if self._buffer else None

    def get(self, timeout: Optional[float] = None) -> StepResult:
        """
        Blocks until a new StepResult is available or until timeout (if provided) expires.
        """
        with self._condition:
            current_version = self._version
            success = self._condition.wait_for(lambda: self._version > current_version, timeout=timeout)
            if not success:
                raise TimeoutError("Timed out while waiting for observation.")
            return self._buffer[-1]

    def snapshot(self) -> Sequence[StepResult]:
        with self._condition:
            return tuple(self._buffer)


class ActionStream:
    """
    Thread-safe storage for the most recent action supplied by the policy. The simulation
    always consumes the most up-to-date action when performing a control step.
    """

    def __init__(self, initial_action: np.ndarray):
        initial_array = np.asarray(initial_action, dtype=np.float32).copy()
        self._lock = threading.Lock()
        self._latest = (time.time(), initial_array)

    def push(self, action: Sequence[float]):
        action_array = np.asarray(action, dtype=self._latest[1].dtype).copy()
        with self._lock:
            self._latest = (time.time(), action_array)

    def latest(self) -> np.ndarray:
        with self._lock:
            return self._latest[1].copy()

    def latest_timestamp(self) -> float:
        with self._lock:
            return self._latest[0]


class AsyncSimulation:
    """
    Runs a robosuite environment in a background thread at (approximately) real-time rate.
    Policies interact with the environment through action and observation streams.
    """

    def __init__(
        self,
        env,
        action_freq: float = 50.0,
        observation_freq: float = 30.0,
        history: int = 1,
        auto_reset: bool = True,
    ):
        if action_freq <= 0:
            raise ValueError("action_freq must be > 0.")
        if observation_freq <= 0:
            raise ValueError("observation_freq must be > 0.")

        self.env = env
        self.action_freq = float(action_freq)
        self.observation_freq = float(observation_freq)
        self.auto_reset = auto_reset

        default_action = self._default_action()
        self.action_stream = ActionStream(initial_action=default_action)
        self.observation_stream = ObservationStream(history=history)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._latest_step: Optional[StepResult] = None

    def _default_action(self) -> np.ndarray:
        try:
            low, high = self.env.action_spec
            low = np.asarray(low, dtype=np.float32)
            zeros = np.zeros_like(low, dtype=np.float32)
            return zeros
        except Exception as exc:
            raise RuntimeError("Unable to determine default action from env.action_spec.") from exc

    def _reset_env(self) -> StepResult:
        observation = self.env.reset()
        try:
            self.env.initialize_time(self.action_freq)
        except Exception as exc:
            logger.debug("initialize_time during reset failed with %s", exc)
        timestamp = time.time()
        info = {"reset": True}
        action = self.action_stream.latest()
        step = StepResult(
            observation=observation,
            reward=0.0,
            done=False,
            info=info,
            action=action,
            timestamp=timestamp,
        )
        self._latest_step = step
        self.observation_stream.publish(step)
        return step

    def _step_once(self) -> StepResult:
        action = self.action_stream.latest()
        observation, reward, done, info = self.env.step(action)
        timestamp = time.time()
        step = StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            action=action,
            timestamp=timestamp,
        )
        self._latest_step = step
        return step

    def _loop(self):
        try:
            self.env.initialize_time(self.action_freq)
        except Exception as exc:
            logger.debug("initialize_time failed with %s. Continuing without reconfiguring control frequency.", exc)

        self._reset_env()

        action_period = 1.0 / self.action_freq
        observation_period = 1.0 / self.observation_freq
        next_action_time = time.perf_counter()
        next_observation_time = next_action_time
        self._started_event.set()

        while not self._stop_event.is_set():
            now = time.perf_counter()
            sleep_time = next_action_time - now
            if sleep_time > 0:
                time.sleep(min(sleep_time, 0.001))
                continue

            step = self._step_once()
            now = time.perf_counter()
            if now >= next_observation_time:
                self.observation_stream.publish(step)
                while next_observation_time <= now:
                    next_observation_time += observation_period

            next_action_time += action_period
            if next_action_time <= now:
                next_action_time = now + action_period

            if step.done:
                if not self.auto_reset:
                    self._stop_event.set()
                    break
                self._reset_env()
                next_action_time = time.perf_counter() + action_period
                next_observation_time = time.perf_counter()

    def start(self, wait: bool = True):
        if self._thread and self._thread.is_alive():
            raise RuntimeError("AsyncSimulation already running.")
        self._stop_event.clear()
        self._started_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="AsyncSimulationThread")
        self._thread.start()
        if wait:
            started = self._started_event.wait(timeout=5.0)
            if not started:
                raise TimeoutError("AsyncSimulation failed to start within 5 seconds.")

    def stop(self, wait: bool = True):
        self._stop_event.set()
        if self._thread and wait:
            self._thread.join(timeout=5.0)
        self._thread = None
        self._started_event.clear()

    def latest_step(self) -> Optional[StepResult]:
        return self._latest_step

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(wait=True)


