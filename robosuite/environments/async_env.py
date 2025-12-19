from __future__ import annotations
import logging
import mujoco
import multiprocessing as mp
import queue
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from functools import cached_property, partial
from multiprocessing.context import SpawnContext
from multiprocessing.queues import Queue as MPQueue
from typing import Any, Callable, Dict, Optional, Sequence, TYPE_CHECKING

import numpy as np
import robosuite as suite

if TYPE_CHECKING:
    import mujoco.viewer
    from robosuite.environments.base import MujocoEnv
    from robosuite.renderers.viewer import MjviewerRenderer
    from multiprocessing.context import SpawnProcess
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


class PeriodicEventGenerator:
    def __init__(self, frequency: float, restart_period: bool = False, start_time: float = 0.0):
        assert frequency > 0.0, "Frequency must be strictly positive."
        assert start_time >= 0.0, "Start time must be non-negative."
        self.period: float = 1.0 / frequency
        self.last_event_time: float = start_time - self.period
        self.restart_period = restart_period

    def is_ready(self, timestamp: float) -> bool:
        return timestamp >= self.last_event_time + self.period

    def register_event(self, timestamp: float):
        if not self.is_ready(timestamp):
            raise ValueError(
                "PeriodicEventGenerator.register_event() called at an invalid time. "
                "Call is_ready() first to check if the event is ready to be registered."
            )

        time_elapsed = timestamp - self.last_event_time
        assert time_elapsed >= 0.0, "Time elapsed must be non-negative."

        self.last_event_time = timestamp

        # Register this event not to the current time, but to the closest previous multiple of period.
        if self.restart_period:
            self.last_event_time -= time_elapsed % self.period

class Queue (MPQueue):
    """
    Multiprocessing queue that supports drop-oldest publish and draining the latest item.
    """

    def publish(self, item: Any):
        """
        Put an item into the queue, dropping the oldest element if the queue is full.
        """
        while True:
            try:
                self.put_nowait(item)
                break
            except queue.Full:
                try:
                    self.get_nowait()
                except queue.Empty:
                    continue

    def drain(self) -> Any:
        """
        Drain all items from the queue and return the latest one.
        """
        latest: Optional[Any] = None
        while True:
            try:
                latest = self.get_nowait()
            except queue.Empty:
                break
        return latest


@dataclass
class SimulationWorkerConf:
    env_factory: Callable[[], MujocoEnv]
    control_freq: float
    observation_freq: float
    visualization_freq: Optional[float]
    target_real_time_rate: float
    reward_freq: float
    start_event: Event
    stop_event: Event
    control_queue: Queue
    observation_queue: Queue
    request_queue: Queue
    reply_queue: Queue

class SimulationWorker:
    conf: SimulationWorkerConf
    env: MujocoEnv
    control_peg: PeriodicEventGenerator
    obs_peg: PeriodicEventGenerator
    reward_peg: PeriodicEventGenerator
    viz_peg: Optional[PeriodicEventGenerator]
    latest_action: np.ndarray
    episode_done: bool
    last_sim_stamp: float
    last_real_stamp: float

    def __init__(self, conf: SimulationWorkerConf):
        self.conf = conf

        # Instantiate environment in the worker process.
        self.env = self.conf.env_factory()

        # Reset the environment
        self.reset()

        # Signal to the parent process that initialization is complete.
        self.conf.start_event.set()

    @property
    def default_action(self) -> np.ndarray:
        return np.zeros_like(self.env.action_spec[0], dtype=np.float32)

    @property
    def sim_time(self) -> float:
        return float(self.env.sim.data.time)

    @property
    def observation_queue(self) -> Queue:
        return self.conf.observation_queue
    
    @property
    def control_queue(self) -> Queue:
        return self.conf.control_queue
    
    @property
    def request_queue(self) -> Queue:
        return self.conf.request_queue
    
    @property
    def reply_queue(self) -> Queue:
        return self.conf.reply_queue
    
    def reset(self):
        self.env.initialize_time(self.conf.control_freq)

        observation = self.env.reset()
        self.observation_queue.drain()
        self.latest_action = self.default_action
        self.control_queue.drain()

        # Reset periodic event generators
        # Create periodic event generators.
        self.control_peg = PeriodicEventGenerator(self.conf.control_freq, restart_period=False)
        self.obs_peg = PeriodicEventGenerator(frequency=self.conf.observation_freq, restart_period=False)
        self.reward_peg = PeriodicEventGenerator(self.conf.reward_freq, restart_period=False)
        if self.conf.visualization_freq is not None:
            self.viz_peg = PeriodicEventGenerator(self.conf.visualization_freq, restart_period=True)
        else:
            self.viz_peg = None

        current_step = StepResult(
            observation=observation,
            reward=0.0,
            done=False,
            info={"reset": True},
            action=self.latest_action.copy(),
            timestamp=time.time(),
        )
        self.observation_queue.publish(current_step)
        self.episode_done = False

    def handle_request(self, request: Request):
        assert isinstance(request, Request), f"Expected Request, got {type(request)}"
        command = request.command

        if command == "reset":
            self.reset()
            next_control_time = time.perf_counter()
            next_observation_time = next_control_time
            self.reply_queue.put(Response(command=command, payload=None))
        elif command == "shutdown":
            self.conf.stop_event.set()
            self.reply_queue.put(Response(command=command, payload=None))
        elif command == "get_action_dim":
            self.reply_queue.put(Response(command=command, payload=self.env.action_dim))
        else:
            raise ValueError(f"Unknown command received by simulation worker: {request.command!r}")

    def is_any_peg_ready(self) -> bool:
        return (
            self.control_peg.is_ready(self.sim_time) or
            self.obs_peg.is_ready(self.sim_time) or
            self.reward_peg.is_ready(self.sim_time)
        )

    def take_env_step(self) -> StepResult:
        observation, reward, done, info = self.env.step(self.latest_action.copy())

        timestamp = time.time()
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
            action=self.latest_action.copy(),
            timestamp=timestamp,
        )

    def update_timestamps(self):
        self.last_sim_stamp = self.sim_time
        self.last_real_stamp = time.perf_counter()

    def sync_time(self):
        curr_real_time = time.perf_counter()
        real_duration = curr_real_time - self.last_real_stamp
        sim_duration = self.sim_time - self.last_sim_stamp

        # Sleep to catch up with simulation time
        target_real_duration = sim_duration / self.conf.target_real_time_rate
        if target_real_duration >= real_duration:
            time.sleep(target_real_duration - real_duration)

        self.update_timestamps()

    def add_overlays_to_viz(self):
        renderer: MjviewerRenderer | None = getattr(self.env, "viewer", None)
        if renderer is None: return
        viewer_handle: mujoco.viewer.Handle = renderer.viewer

        # TODO (Sid): compute real time rate
        observed_rtr = 1.0
        viewer_handle.set_texts([
            (None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "Async. simulation time", f"{self.sim_time:.3f}s"),
            (None, mujoco.mjtGridPos.mjGRID_TOPRIGHT, "Real-time rate", f"{observed_rtr:.3f}")
        ])

    def run(self):
        self.update_timestamps()

        # Main worker loop
        while True:
            # Check for requests from the parent process.
            try:
                # Requests are blocking until there is a reply on thr reply queue.
                # So there will only be one request at a time.
                request = self.conf.request_queue.get_nowait()

                # If a request is succesfully received, handle it.
                self.handle_request(request)
            except queue.Empty:
                pass

            # Stop running the loop if the stop event is set.
            if self.conf.stop_event.is_set():
                break

            # Take a step in the environment.
            current_step = self.take_env_step()

            # Check if any of the periodic event generators are ready. 
            # If any of them is ready, then we need to synchronize real time with simulation time.
            if self.is_any_peg_ready():
                self.sync_time()

            # Update the latest action from the control queue.
            if self.control_peg.is_ready(self.sim_time):
                if (drained_action := self.control_queue.drain()) is not None:
                    self.latest_action = np.array(drained_action, dtype=np.float32)
                self.control_peg.register_event(self.sim_time)
        
            # Publish observation.
            if self.obs_peg.is_ready(self.sim_time):
                self.observation_queue.publish(current_step)
                self.obs_peg.register_event(self.sim_time)
    
            # Update visualization.
            if self.viz_peg is not None and self.viz_peg.is_ready(self.sim_time):
                self.add_overlays_to_viz()
                self.env.render()
                self.viz_peg.register_event(self.sim_time)

            if current_step.done:
                self.observation_queue.publish(current_step)

        self.env.close()

def _simulation_worker(conf: SimulationWorkerConf):
    worker = SimulationWorker(conf)
    worker.run()

def _simulation_worker_old(
    env_factory: Callable[[], MujocoEnv],
    control_freq: float,
    observation_freq: float,
    visualization_freq: Optional[float],
    reward_freq: float,
    start_event: Event,
    stop_event: Event,
    control_queue: Queue,
    observation_queue: Queue,
    request_queue: Queue,
    reply_queue: Queue,
):
    # Instantiate environment in the worker process.
    env = env_factory()

    def reset_env():
        env.initialize_time(control_freq)
        observation = env.reset()
        observation_queue.drain()
        default_action = np.zeros_like(env.action_spec[0], dtype=np.float32)
        latest_action = default_action
        control_queue.drain()

        # Initialize periodic event generators
        control_peg = PeriodicEventGenerator(control_freq, restart_period=False)
        obs_peg = PeriodicEventGenerator(frequency=observation_freq, restart_period=False)
        reward_peg = PeriodicEventGenerator(reward_freq, restart_period=False)
        if visualization_freq is not None:
            viz_peg = PeriodicEventGenerator(visualization_freq, restart_period=True)
        else:
            viz_peg = None

        current_step = StepResult(
            observation=observation,
            reward=0.0,
            done=False,
            info={"reset": True},
            action=latest_action.copy(),
            timestamp=time.time(),
        )
        observation_queue.publish(current_step)

        sim_time: float = env.sim.data.time
        episode_done: bool = False

        return sim_time, latest_action, current_step, episode_done, control_peg, obs_peg, reward_peg, viz_peg

    # Reset the environment
    (
        sim_time,
        latest_action,
        current_step,
        episode_done,
        control_peg,
        observation_peg,
        reward_peg,
        visualization_peg,
    ) = reset_env()

    start_event.set()

    control_period = 1.0 / control_freq
    observation_period = 1.0 / observation_freq
    next_control_time = time.perf_counter()
    next_observation_time = next_control_time

    while True:
        # Check for requests from the parent process.
        try:
            # Requests are blocking until there is a reply on thr reply queue.
            # So there will only be one request at a time.
            request: Request = request_queue.get_nowait()

            assert isinstance(request, Request), f"Expected Request, got {type(request)}"
            command = request.command

            if command == "reset":
                (
                    sim_time,
                    latest_action,
                    current_step,
                    episode_done,
                    control_peg,
                    observation_peg,
                    reward_peg,
                    visualization_peg,
                ) = reset_env()
                next_control_time = time.perf_counter()
                next_observation_time = next_control_time
                reply_queue.put(Response(command=command, payload=None))
            elif command == "shutdown":
                stop_event.set()
                reply_queue.put(Response(command=command, payload=None))
                break
            elif command == "get_action_dim":
                reply_queue.put(Response(command=command, payload=env.action_dim))
            else:
                raise ValueError(f"Unknown command received by simulation worker: {request.command!r}")

        except queue.Empty:
            pass

        # Take a step in the environment.
        # observation, reward, done, info = env.step(latest_action.copy())
        # sim_time = env.sim.data.time
        # TODO (Sid): We're continuing from here.

        # ==========================================================================================

        now = time.perf_counter()
        sleep_time = next_control_time - now
        if sleep_time > 0:
            if stop_event.wait(timeout=min(sleep_time, 0.002)):
                break
            continue

        # Update the latest action from the control queue.
        if (drained_action := control_queue.drain()) is not None:
            latest_action = np.array(drained_action, dtype=np.float32)

        # Take a step in the environment.
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
            observation_queue.publish(current_step)
            while next_observation_time <= now:
                next_observation_time += observation_period

        next_control_time += control_period
        if next_control_time <= now:
            next_control_time = now + control_period

        if done:
            episode_done = True
            if not should_publish:
                observation_queue.publish(current_step)
            next_control_time = time.perf_counter()
            next_observation_time = next_control_time

    env.close()


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


class ControlStream:
    """
    Interface for pushing actions (controls) to the asynchronous simulation process.
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
        self._queue.publish(action_array)

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
        control_freq: float = 50.0,
        observation_freq: float = 30.0,
        visualization_freq: Optional[float] = 20.0,
        target_real_time_rate: float = 1.0,
        reward_freq: float = 10.0,
        history: int = 1,
        ctx: Optional[SpawnContext] = None,
    ):
        if control_freq <= 0:
            raise ValueError("control_freq must be > 0.")
        if observation_freq <= 0:
            raise ValueError("observation_freq must be > 0.")

        self._ctx = ctx or mp.get_context("spawn")
        self._start_event = self._ctx.Event()
        self._stop_event = self._ctx.Event()
        self._control_queue = Queue(maxsize=8, ctx=self._ctx)
        self._observation_queue = Queue(maxsize=max(1, history), ctx=self._ctx)
        self._request_queue = Queue(maxsize=1, ctx=self._ctx)
        self._reply_queue = Queue(maxsize=1, ctx=self._ctx)
        self._process: Optional[SpawnProcess] = None

        self.control_stream = ControlStream(self._control_queue)
        self.observation_stream = ObservationStream(self._observation_queue, history=history)

        self.worker_conf = SimulationWorkerConf(
            env_factory,
            control_freq,
            observation_freq,
            visualization_freq,
            target_real_time_rate,
            reward_freq,
            self._start_event,
            self._stop_event,
            self._control_queue,
            self._observation_queue,
            self._request_queue,
            self._reply_queue
        )

    def start(self, wait: bool = True):
        if self._process and self._process.is_alive():
            raise RuntimeError("AsyncSimulation already running.")

        self._start_event.clear()
        self._stop_event.clear()

        self._process = self._ctx.Process(
            target=_simulation_worker,
            name="AsyncSimulationProcess",
            daemon=True,
            args=(self.worker_conf,),
        )
        self._process.start()

        if wait:
            started = self._start_event.wait(timeout=10.0)
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
        self._start_event.clear()

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
            response = self._reply_queue.get(block=True, timeout=timeout)
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
    control_freq: float = 50.0,
    observation_freq: float = 30.0,
    history: int = 1,
    **kwargs,
) -> AsyncSimulation:
    """
    Instantiates an asynchronous simluation of a robosuite environment.
    Args:
        env_name (str): Name of the robosuite environment to initialize
        *args: Additional arguments to pass to the specific environment class initializer
        control_freq (float): The frequency of the control stream in Hz.
        observation_freq (float): The frequency of the observation stream in Hz.
        history (int): The number of steps to store in the observation stream.
        **kwargs: Additional arguments to pass to the specific environment class initializer
    Returns:
        AsyncSimulation: Asynchronous simulation of the robosuite environment.
    """
    return AsyncSimulation(
        env_factory = partial(suite.make, env_name, *args, control_freq = control_freq, **kwargs),
        control_freq = control_freq,
        observation_freq = observation_freq,
        history = history,
    )
