import logging
import multiprocessing as mp
import queue
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import Any, Callable, Dict, Optional, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robosuite.environments.base import MujocoEnv
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event

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


def _default_action_from_env(env) -> np.ndarray:
    low, _ = env.action_spec
    low = np.asarray(low, dtype=np.float32)
    return np.zeros_like(low, dtype=np.float32)


def _publish_step(queue_obj: "Queue", step: StepResult):
    while True:
        try:
            queue_obj.put_nowait(step)
            break
        except queue.Full:
            try:
                queue_obj.get_nowait()
            except queue.Empty:
                break


def _simulation_worker(
    env_factory: Callable[[], "MujocoEnv"],
    action_freq: float,
    observation_freq: float,
    stop_event: "Event",
    started_event: "Event",
    action_queue: "Queue",
    observation_queue: "Queue",
    command_queue: "Queue",
):
    env = env_factory()

    try:
        env.initialize_time(action_freq)
    except Exception as exc:
        logger.debug("initialize_time failed with %s. Continuing with env defaults.", exc)

    default_action = _default_action_from_env(env)
    latest_action = default_action.copy()
    current_step: Optional[StepResult] = None
    episode_done = False

    def reset_env():
        nonlocal latest_action, current_step
        try:
            env.initialize_time(action_freq)
        except Exception as exc:
            logger.debug("initialize_time during reset failed with %s", exc)
        observation = env.reset()
        latest_action = default_action.copy()
        current_step = StepResult(
            observation=observation,
            reward=0.0,
            done=False,
            info={"reset": True},
            action=latest_action.copy(),
            timestamp=time.time(),
        )
        _publish_step(observation_queue, current_step)

    def drain_actions():
        nonlocal latest_action
        while True:
            try:
                candidate = action_queue.get_nowait()
                latest_action = np.asarray(candidate, dtype=np.float32).copy()
            except queue.Empty:
                break

    reset_env()
    started_event.set()

    action_period = 1.0 / action_freq
    observation_period = 1.0 / observation_freq
    next_action_time = time.perf_counter()
    next_observation_time = next_action_time

    while not stop_event.is_set():
        while True:
            try:
                cmd = command_queue.get_nowait()
            except queue.Empty:
                break
            if cmd == "reset":
                reset_env()
                episode_done = False
                next_action_time = time.perf_counter()
                next_observation_time = next_action_time
            elif cmd == "shutdown":
                stop_event.set()
                break
        if stop_event.is_set():
            break

        if episode_done:
            stop_event.wait(timeout=0.01)
            continue

        now = time.perf_counter()
        sleep_time = next_action_time - now
        if sleep_time > 0:
            if stop_event.wait(timeout=min(sleep_time, 0.002)):
                break
            continue

        drain_actions()
        observation, reward, done, info = env.step(latest_action.copy())
        timestamp = time.time()
        current_step = StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            action=latest_action.copy(),
            timestamp=timestamp,
        )

        now = time.perf_counter()
        should_publish = now >= next_observation_time or done
        if should_publish:
            _publish_step(observation_queue, current_step)
            while next_observation_time <= now:
                next_observation_time += observation_period

        next_action_time += action_period
        if next_action_time <= now:
            next_action_time = now + action_period

        if done:
            episode_done = True
            if not should_publish:
                _publish_step(observation_queue, current_step)
            next_action_time = time.perf_counter()
            next_observation_time = next_action_time

    try:
        env.close()
    except Exception:
        pass


class ObservationStream:
    """
    Interface for consuming observations produced by the asynchronous simulation process.
    """

    def __init__(self, queue_obj: "Queue", history: int = 1):
        if history <= 0:
            raise ValueError("ObservationStream history must be a positive integer.")
        self._queue = queue_obj
        self._history = deque(maxlen=history)
        self._lock = threading.Lock()

    def _drain(self, block: bool, timeout: Optional[float]) -> Optional[StepResult]:
        first = True
        while True:
            try:
                if block and first:
                    if timeout is None:
                        item = self._queue.get(block=True)
                    else:
                        item = self._queue.get(block=True, timeout=timeout)
                else:
                    item = self._queue.get_nowait()
                with self._lock:
                    self._history.append(item)
                first = False
                block = False
            except queue.Empty:
                break
        with self._lock:
            return self._history[-1] if self._history else None

    def latest(self) -> Optional[StepResult]:
        """
        Returns the latest step without blocking.
        If the queue is empty, returns None.
        """
        return self._drain(block=False, timeout=None)

    def get(self, timeout: Optional[float] = None) -> StepResult:
        """
        Returns the latest step with blocking.
        If the queue is empty, waits for the step to be available.
        """
        result = self._drain(block=True, timeout=timeout)
        if result is None:
            raise TimeoutError("Timed out while waiting for observation.")
        return result

    def snapshot(self) -> Sequence[StepResult]:
        self._drain(block=False, timeout=None)
        with self._lock:
            return tuple(self._history)


class ActionStream:
    """
    Interface for pushing actions to the asynchronous simulation process.
    """

    def __init__(self, queue_obj: "Queue"):
        self._queue = queue_obj
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0

    def push(self, action: Sequence[float]):
        action_array = np.asarray(action, dtype=np.float32).copy()
        with self._lock:
            self._latest = action_array
            self._latest_timestamp = time.time()
        while True:
            try:
                self._queue.put_nowait(action_array)
                break
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    def latest(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def latest_timestamp(self) -> float:
        with self._lock:
            return self._latest_timestamp


class AsyncSimulation:
    """
    Runs a robosuite environment in a separate process at approximately real-time rate.
    The parent process (typically running the policy) interacts with the simulation through
    action and observation streams.
    """

    def __init__(
        self,
        env_factory: Callable[[], "MujocoEnv"],
        action_freq: float = 50.0,
        observation_freq: float = 30.0,
        history: int = 1,
        ctx: Optional[BaseContext] = None,
    ):
        if action_freq <= 0:
            raise ValueError("action_freq must be > 0.")
        if observation_freq <= 0:
            raise ValueError("observation_freq must be > 0.")

        self.env_factory = env_factory
        self.action_freq = float(action_freq)
        self.observation_freq = float(observation_freq)
        self._ctx = ctx or mp.get_context("spawn")
        self._stop_event = self._ctx.Event()
        self._started_event = self._ctx.Event()
        self._action_queue = self._ctx.Queue(maxsize=8)
        self._observation_queue = self._ctx.Queue(maxsize=max(1, history))
        self._command_queue = self._ctx.Queue()
        self._process: Optional[mp.Process] = None

        self.action_stream = ActionStream(self._action_queue)
        self.observation_stream = ObservationStream(self._observation_queue, history=history)

    def start(self, wait: bool = True):
        if self._process and self._process.is_alive():
            raise RuntimeError("AsyncSimulation already running.")

        self._stop_event.clear()
        self._started_event.clear()

        self._process = self._ctx.Process(
            target=_simulation_worker,
            name="AsyncSimulationProcess",
            daemon=True,
            args=(
                self.env_factory,
                self.action_freq,
                self.observation_freq,
                self._stop_event,
                self._started_event,
                self._action_queue,
                self._observation_queue,
                self._command_queue,
            ),
        )
        self._process.start()

        if wait:
            started = self._started_event.wait(timeout=10.0)
            if not started:
                self.stop(wait=False)
                raise TimeoutError("AsyncSimulation failed to start within 10 seconds.")

    def stop(self, wait: bool = True):
        if self._process is None:
            return
        if not self._stop_event.is_set():
            self._stop_event.set()
        try:
            self._command_queue.put_nowait("shutdown")
        except queue.Full:
            pass
        if wait and self._process.is_alive():
            self._process.join(timeout=10.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)
        self._process = None
        self._started_event.clear()

    def reset(self):
        if not self.is_running():
            raise RuntimeError("AsyncSimulation is not running.")
        self._command_queue.put("reset")

    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def latest_step(self) -> Optional[StepResult]:
        return self.observation_stream.latest()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(wait=True)


