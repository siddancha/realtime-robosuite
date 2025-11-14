from __future__ import annotations
import logging
import multiprocessing as mp
import queue
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from functools import cached_property, partial
from multiprocessing.context import BaseContext
from typing import Any, Callable, Dict, Optional, Sequence, TYPE_CHECKING

import numpy as np
import robosuite as suite

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


@dataclass(frozen=True)
class Request:
    command: str
    payload: Any = None


@dataclass(frozen=True)
class Response:
    command: str
    payload: Any = None
    error: Optional[str] = None


def _simulation_worker(
    env_factory: Callable[[], MujocoEnv],
    action_freq: float,
    observation_freq: float,
    stop_event: Event,
    started_event: Event,
    action_queue: Queue,
    observation_queue: Queue,
    request_queue: Queue,
    reply_queue: Queue,
):
    # Instantiate environment in the worker process.
    env = env_factory()

    default_action = np.zeros_like(env.action_spec[0], dtype=np.float32)
    latest_action = default_action.copy()
    current_step: Optional[StepResult] = None
    episode_done = False

    def publish_observation(step: StepResult):
        while True:
            try:
                observation_queue.put_nowait(step)
                break
            except queue.Full:
                try:
                    observation_queue.get_nowait()
                except queue.Empty:
                    break

    def reset_env():
        nonlocal latest_action, current_step
        env.initialize_time(action_freq)
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
        publish_observation(current_step)

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
                request: Request = request_queue.get_nowait()
            except queue.Empty:
                break

            assert isinstance(request, Request), f"Expected Request, got {type(request)}"
            command = request.command

            if command == "reset":
                reset_env()
                episode_done = False
                next_action_time = time.perf_counter()
                next_observation_time = next_action_time
                reply_queue.put(Response(command=command, payload=None))
            elif command == "shutdown":
                stop_event.set()
                reply_queue.put(Response(command=command, payload=None))
                break
            elif command == "get_action_dim":
                reply_queue.put(Response(command=command, payload=env.action_dim))
            else:
                raise ValueError(f"Unknown command received by simulation worker: {request.command!r}")
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
            publish_observation(current_step)
            while next_observation_time <= now:
                next_observation_time += observation_period

        next_action_time += action_period
        if next_action_time <= now:
            next_action_time = now + action_period

        if done:
            episode_done = True
            if not should_publish:
                publish_observation(current_step)
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

    def __init__(self, queue_obj: Queue, history: int = 1):
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

    def __init__(self, queue_obj: Queue):
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
        env_factory: Callable[[], MujocoEnv],
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
        self._request_queue = self._ctx.Queue(maxsize=1)
        self._reply_queue = self._ctx.Queue(maxsize=1)
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
                self._request_queue,
                self._reply_queue,
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
        if self.is_running():
            try:
                self._send_request("shutdown", timeout=2.0)
            except Exception:
                pass
        if not self._stop_event.is_set():
            self._stop_event.set()
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
        self._send_request("reset", timeout=2.0)

    def _send_request(
        self,
        command: str,
        payload: Any = None,
        timeout: Optional[float] = None,
    ) -> Any:
        if not self.is_running():
            raise RuntimeError(f"AsyncSimulation is not running; cannot send {command!r} command.")

        request = Request(command=command, payload=payload)
        self._request_queue.put(request)
        try:
            response = self._reply_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise RuntimeError(f"Timed out waiting for response to command {command!r}.") from exc

        if not isinstance(response, Response):
            raise RuntimeError(
                f"Received unknown message type from asynchronous simulation: {response!r}"
            )

        if response.command != command:
            raise RuntimeError(
                f"Received mismatched response '{response.command}' for command '{command}'."
            )

        if response.error is not None:
            raise RuntimeError(
                f"Command '{command}' failed in asynchronous simulation: {response.error}"
            )

        return response.payload

    @cached_property
    def action_dim(self) -> int:
        if not self.is_running():
            raise RuntimeError("AsyncSimulation must be running to query action_dim.")
        result = self._send_request("get_action_dim", timeout=2.0)
        return int(result)

    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def latest_step(self) -> Optional[StepResult]:
        return self.observation_stream.latest()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(wait=True)


def make_async(
    env_name: str,
    *args,
    action_freq: float = 50.0,
    observation_freq: float = 30.0,
    history: int = 1,
    **kwargs,
) -> AsyncSimulation:
    """
    Instantiates an asynchronous simluation of a robosuite environment.
    Args:
        env_name (str): Name of the robosuite environment to initialize
        *args: Additional arguments to pass to the specific environment class initializer
        action_freq (float): The frequency of the action stream in Hz.
        observation_freq (float): The frequency of the observation stream in Hz.
        history (int): The number of steps to store in the observation stream.
        **kwargs: Additional arguments to pass to the specific environment class initializer
    Returns:
        AsyncSimulation: Asynchronous simulation of the robosuite environment.
    """
    return AsyncSimulation(
        env_factory = partial["MujocoEnv"](suite.make, env_name, *args, **kwargs),
        action_freq = action_freq,
        observation_freq = observation_freq,
        history = history,
    )
