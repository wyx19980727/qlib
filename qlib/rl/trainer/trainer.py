# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import collections
import copy
import threading
import time
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty
from typing import Any, Dict, Iterable, List, Optional, OrderedDict, Sequence, TypeVar, cast

import numpy as np
import torch

from qlib.log import get_module_logger
from qlib.rl.simulator import InitialStateType
from qlib.rl.utils import EnvWrapper, FiniteEnvType, LogBuffer, LogCollector, LogLevel, LogWriter, vectorize_env
from qlib.rl.utils.finite_env import FiniteVectorEnv
from qlib.typehint import Literal

from multiprocessing import Value as MpValue
from multiprocessing.queues import Queue as MpQueue

from .callbacks import Callback
from .vessel import TrainingVesselBase

_ENV_ATTR = "env"


def _clear_env_ref(obj: Any) -> None:
    """Remove .env (weakref.proxy) from obj to make it picklable.

    weakref.proxy objects cannot be serialized by cloudpickle, which blocks
    subproc/shmem vector envs from sending the env_factory closure to child
    processes.  Child processes will set .env again via EnvWrapper.__init__.
    """
    if obj is not None:
        try:
            if hasattr(obj, _ENV_ATTR):
                delattr(obj, _ENV_ATTR)
        except (AttributeError, ReferenceError):
            pass


class _MpQueueIterable:
    """Fork-safe wrapper around a multiprocessing.Queue for seed iteration.

    Reads directly from the underlying mp.Queue, avoiding the DataQueue's
    daemon-thread-based activation which deadlocks on fork().
    """

    def __init__(self, queue: MpQueue, done_flag: Any, data_queue: Any = None) -> None:
        self._queue = queue
        self._done = done_flag
        self._empty_count = 0
        # Keep a reference to the source DataQueue alive so that its
        # __del__ (which calls cleanup() and increments _done) is not
        # triggered while training is running.
        self._data_queue_ref = data_queue

    def __iter__(self) -> "_MpQueueIterable":
        return self

    def cleanup(self) -> None:
        """Signal the producer to stop and release resources."""
        with self._done.get_lock():
            if self._done.value == 0:
                self._done.value = 1
        self._data_queue_ref = None

    def __next__(self) -> Any:
        timeout = 5.0
        while True:
            try:
                return self._queue.get(timeout=timeout)
            except Empty:
                self._empty_count += 1
                if self._done.value:
                    _logger.info(
                        "Seed queue exhausted after %d empty polls.",
                        self._empty_count,
                    )
                    raise StopIteration
                timeout = 0.5


_logger = get_module_logger(__name__)


def _start_simple_producer(
    dataset: Sequence[Any],
    queue: MpQueue,
    done_flag: Any,
    shuffle: bool = True,
    repeat: int = -1,
    seed: int = 42,
) -> threading.Thread:
    """Start a daemon thread that produces items from *dataset* into *queue*.

    Unlike ``DataQueue._producer``, this function does **not** use
    ``torch.utils.data.DataLoader``, because DataLoader + daemon thread is
    unreliable on Python 3.11+ (fork-from-daemon-thread deadlock).

    Parameters
    ----------
    dataset
        Sequence of items to iterate over.
    queue
        ``multiprocessing.Queue`` to push items into.
    done_flag
        ``multiprocessing.Value('i', 0)``; the producer sets it to 1 on exit.
    shuffle
        If true, items are emitted in a random order each epoch.
    repeat
        Number of epochs.  -1 means infinite.
    seed
        Seed for independent RandomState.  Keeps the producer's RNG
        isolated from the main thread's global ``numpy.random``.
    """
    rng = np.random.RandomState(seed)

    def _produce() -> None:
        n = len(dataset)
        if n == 0:
            _logger.warning("Simple producer: dataset is empty, nothing to produce.")
            with done_flag.get_lock():
                done_flag.value = 1
            return
        epoch = 0
        try:
            while repeat == -1 or epoch < repeat:
                indices = rng.permutation(n).tolist() if shuffle else list(range(n))
                for idx in indices:
                    if done_flag.value:
                        return
                    queue.put(dataset[idx])
                epoch += 1
        finally:
            with done_flag.get_lock():
                done_flag.value = 1

    t = threading.Thread(target=_produce, daemon=True)
    t.start()
    return t


T = TypeVar("T")


class Trainer:
    """
    Utility to train a policy on a particular task.

    Different from traditional DL trainer, the iteration of this trainer is "collect",
    rather than "epoch", or "mini-batch".
    In each collect, :class:`Collector` collects a number of policy-env interactions, and accumulates
    them into a replay buffer. This buffer is used as the "data" to train the policy.
    At the end of each collect, the policy is *updated* several times.

    The API has some resemblence with `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/>`__,
    but it's essentially different because this trainer is built for RL applications, and thus
    most configurations are under RL context.
    We are still looking for ways to incorporate existing trainer libraries, because it looks like
    big efforts to build a trainer as powerful as those libraries, and also, that's not our primary goal.

    It's essentially different
    `tianshou's built-in trainers <https://tianshou.readthedocs.io/en/master/api/tianshou.trainer.html>`__,
    as it's far much more complicated than that.

    Parameters
    ----------
    max_iters
        Maximum iterations before stopping.
    val_every_n_iters
        Perform validation every n iterations (i.e., training collects).
    logger
        Logger to record the backtest results. Logger must be present because
        without logger, all information will be lost.
    finite_env_type
        Type of finite env implementation.
    concurrency
        Parallel workers.
    fast_dev_run
        Create a subset for debugging.
        How this is implemented depends on the implementation of training vessel.
        For :class:`~qlib.rl.vessel.TrainingVessel`, if greater than zero,
        a random subset sized ``fast_dev_run`` will be used
        instead of ``train_initial_states`` and ``val_initial_states``.
    """

    should_stop: bool
    """Set to stop the training."""

    metrics: dict
    """Numeric metrics of produced in train/val/test.
    In the middle of training / validation, metrics will be of the latest episode.
    When each iteration of training / validation finishes, metrics will be the aggregation
    of all episodes encountered in this iteration.

    Cleared on every new iteration of training.

    In fit, validation metrics will be prefixed with ``val/``.
    """

    current_iter: int
    """Current iteration (collect) of training."""

    loggers: List[LogWriter]
    """A list of log writers."""

    def __init__(
        self,
        *,
        max_iters: int | None = None,
        val_every_n_iters: int | None = None,
        loggers: LogWriter | List[LogWriter] | None = None,
        callbacks: List[Callback] | None = None,
        finite_env_type: FiniteEnvType = "subproc",
        concurrency: int = 2,
        fast_dev_run: int | None = None,
    ):
        self.max_iters = max_iters
        self.val_every_n_iters = val_every_n_iters

        if isinstance(loggers, list):
            self.loggers = loggers
        elif isinstance(loggers, LogWriter):
            self.loggers = [loggers]
        else:
            self.loggers = []

        self.loggers.append(LogBuffer(self._metrics_callback, loglevel=self._min_loglevel()))

        self.callbacks: List[Callback] = callbacks if callbacks is not None else []
        self.finite_env_type = finite_env_type
        self.concurrency = concurrency
        self.fast_dev_run = fast_dev_run
        self._fit_start_time: float | None = None

        self.current_stage: Literal["train", "val", "test"] = "train"

        self.vessel: TrainingVesselBase = cast(TrainingVesselBase, None)

    def initialize(self):
        """Initialize the whole training process.

        The states here should be synchronized with state_dict.
        """
        self.should_stop = False
        self.current_iter = 0
        self.current_episode = 0
        self.current_stage = "train"

    def initialize_iter(self):
        """Initialize one iteration / collect."""
        self.metrics = {}

    def state_dict(self) -> dict:
        """Putting every states of current training into a dict, at best effort.

        It doesn't try to handle all the possible kinds of states in the middle of one training collect.
        For most cases at the end of each iteration, things should be usually correct.

        Note that it's also intended behavior that replay buffer data in the collector will be lost.
        """
        return {
            "vessel": self.vessel.state_dict(),
            "callbacks": {name: callback.state_dict() for name, callback in self.named_callbacks().items()},
            "loggers": {name: logger.state_dict() for name, logger in self.named_loggers().items()},
            "should_stop": self.should_stop,
            "current_iter": self.current_iter,
            "current_episode": self.current_episode,
            "current_stage": self.current_stage,
            "metrics": self.metrics,
        }

    @staticmethod
    def get_policy_state_dict(ckpt_path: Path) -> OrderedDict:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "vessel" in state_dict:
            state_dict = state_dict["vessel"]["policy"]
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Load all states into current trainer."""
        self.vessel.load_state_dict(state_dict["vessel"])
        for name, callback in self.named_callbacks().items():
            callback.load_state_dict(state_dict["callbacks"][name])
        for name, logger in self.named_loggers().items():
            logger.load_state_dict(state_dict["loggers"][name])
        self.should_stop = state_dict["should_stop"]
        self.current_iter = state_dict["current_iter"]
        self.current_episode = state_dict["current_episode"]
        self.current_stage = state_dict["current_stage"]
        self.metrics = state_dict["metrics"]

    def named_callbacks(self) -> Dict[str, Callback]:
        """Retrieve a collection of callbacks where each one has a name.
        Useful when saving checkpoints.
        """
        return _named_collection(self.callbacks)

    def named_loggers(self) -> Dict[str, LogWriter]:
        """Retrieve a collection of loggers where each one has a name.
        Useful when saving checkpoints.
        """
        return _named_collection(self.loggers)

    def fit(self, vessel: TrainingVesselBase, ckpt_path: Path | None = None) -> None:
        """Train the RL policy upon the defined simulator.

        Parameters
        ----------
        vessel
            A bundle of all elements used in training.
        ckpt_path
            Load a pre-trained / paused training checkpoint.
        """
        self.vessel = vessel
        vessel.assign_trainer(self)

        if ckpt_path is not None:
            _logger.info("Resuming states from %s", str(ckpt_path))
            self.load_state_dict(torch.load(ckpt_path, weights_only=False))
        else:
            self.initialize()

        self._call_callback_hooks("on_fit_start")
        self._fit_start_time = time.time()

        while not self.should_stop:
            elapsed = time.time() - self._fit_start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            if self.current_iter > 0:
                eta_per_iter = elapsed / self.current_iter
                eta_remaining = eta_per_iter * (self.max_iters - self.current_iter)
                eta_str = str(timedelta(seconds=int(eta_remaining)))
                time_msg = f" | elapsed: {elapsed_str} | ETA: {eta_str}"
            else:
                time_msg = f" | elapsed: {elapsed_str}"

            msg = f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\tTrain iteration {self.current_iter + 1}/{self.max_iters}{time_msg}"
            print(msg, flush=True)
            _logger.info(msg)

            self.initialize_iter()

            self._call_callback_hooks("on_iter_start")

            self.current_stage = "train"
            self._call_callback_hooks("on_train_start")

            # TODO
            # Add a feature that supports reloading the training environment every few iterations.
            iterator = vessel.train_seed_iterator()
            vector_env = None
            try:
                vector_env = self.venv_from_iterator(iterator)
                self.vessel.train(vector_env)
            finally:
                if vector_env is not None:
                    del vector_env  # FIXME: Explicitly delete this object to avoid memory leak.
                if hasattr(iterator, "cleanup"):
                    iterator.cleanup()

            self._call_callback_hooks("on_train_end")

            if self.val_every_n_iters is not None and (self.current_iter + 1) % self.val_every_n_iters == 0:
                # Implementation of validation loop
                self.current_stage = "val"
                self._call_callback_hooks("on_validate_start")
                print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t---- Validation ----", flush=True)
                iterator = vessel.val_seed_iterator()
                vector_env = None
                try:
                    vector_env = self.venv_from_iterator(iterator)
                    self.vessel.validate(vector_env)
                finally:
                    if vector_env is not None:
                        del vector_env  # FIXME: Explicitly delete this object to avoid memory leak.
                    if hasattr(iterator, "cleanup"):
                        iterator.cleanup()

                self._call_callback_hooks("on_validate_end")

            # This iteration is considered complete.
            # Bumping the current iteration counter.
            self.current_iter += 1

            if self.max_iters is not None and self.current_iter >= self.max_iters:
                self.should_stop = True

            self._call_callback_hooks("on_iter_end")

        self._call_callback_hooks("on_fit_end")

    def test(self, vessel: TrainingVesselBase) -> None:
        """Test the RL policy against the simulator.

        The simulator will be fed with data generated in ``test_seed_iterator``.

        Parameters
        ----------
        vessel
            A bundle of all related elements.
        """
        self.vessel = vessel
        vessel.assign_trainer(self)

        self.initialize_iter()

        self.current_stage = "test"
        self._call_callback_hooks("on_test_start")
        iterator = vessel.test_seed_iterator()
        vector_env = None
        try:
            vector_env = self.venv_from_iterator(iterator)
            self.vessel.test(vector_env)
        finally:
            if vector_env is not None:
                del vector_env  # FIXME: Explicitly delete this object to avoid memory leak.
            if hasattr(iterator, "cleanup"):
                iterator.cleanup()
        self._call_callback_hooks("on_test_end")

    def venv_from_iterator(self, iterator: Iterable[InitialStateType]) -> FiniteVectorEnv:
        """Create a vectorized environment from iterator and the training vessel."""

        # For subproc/shmem: DataQueue's daemon thread + fork() causes deadlocks.
        # Instead, we bypass DataQueue and wrap its underlying mp.Queue directly
        # for the children. The producer thread is activated AFTER fork.
        #
        # CRITICAL: DataLoader with num_workers>0 inside a daemon thread spawns
        # subprocesses from the daemon thread, which silently fails on Python 3.11+.
        # Force producer_num_workers=0 to avoid this.
        _data_queue = None  # type: Any
        if self.finite_env_type != "dummy":
            if hasattr(iterator, "_queue"):
                _data_queue = iterator
                _data_queue.producer_num_workers = 0
                _logger.info(
                    "DataQueue producer_num_workers forced to 0 for %s env (daemon thread safety).",
                    self.finite_env_type,
                )
                iterator = _MpQueueIterable(_data_queue._queue, _data_queue._done, _data_queue)

            # weakref.proxy objects (stored as .env on interpreter/reward by EnvWrapper)
            # are NOT picklable - clear before factory creation.
            _clear_env_ref(self.vessel.state_interpreter)
            _clear_env_ref(self.vessel.action_interpreter)
            _clear_env_ref(self.vessel.reward)
        else:
            # dummy mode: activate DataQueue before env_factory (single-process)
            if hasattr(iterator, "activate"):
                iterator.activate()

        def env_factory():
            if self.finite_env_type == "dummy":
                state = copy.deepcopy(self.vessel.state_interpreter)
                action = copy.deepcopy(self.vessel.action_interpreter)
                rew = copy.deepcopy(self.vessel.reward)
            else:
                state = self.vessel.state_interpreter
                action = self.vessel.action_interpreter
                rew = self.vessel.reward

            return EnvWrapper(
                self.vessel.simulator_fn,
                state,
                action,
                iterator,
                rew,
                logger=LogCollector(min_loglevel=self._min_loglevel()),
            )

        vec_env = vectorize_env(
            env_factory,
            self.finite_env_type,
            self.concurrency,
            self.loggers,
        )

        # Clean up after vectorize_env:
        #   - shmem creates dummy env in parent (re-injects .env)
        #   - subproc needs DataQueue producer started after fork
        if self.finite_env_type != "dummy":
            _clear_env_ref(self.vessel.state_interpreter)
            _clear_env_ref(self.vessel.action_interpreter)
            _clear_env_ref(self.vessel.reward)
            if _data_queue is not None:
                # Do NOT call _data_queue.activate() here – its _producer
                # uses torch.utils.data.DataLoader which is unreliable inside
                # a daemon thread on Python 3.11+.  Instead we start a simple
                # numpy-based producer that never uses DataLoader.
                _start_simple_producer(
                    dataset=_data_queue.dataset,
                    queue=_data_queue._queue,
                    done_flag=_data_queue._done,
                    shuffle=_data_queue.shuffle,
                    repeat=_data_queue.repeat,
                    seed=42,
                )
                _logger.info(
                    "Started simple numpy-based producer for %s env (dataset size=%d).",
                    self.finite_env_type,
                    len(_data_queue.dataset),
                )

        return vec_env

    def _metrics_callback(self, on_episode: bool, on_collect: bool, log_buffer: LogBuffer) -> None:
        if on_episode:
            # Update the global counter.
            self.current_episode = log_buffer.global_episode
            metrics = log_buffer.episode_metrics()
        elif on_collect:
            # Update the latest metrics.
            metrics = log_buffer.collect_metrics()
        if self.current_stage == "val":
            metrics = {"val/" + name: value for name, value in metrics.items()}
        self.metrics.update(metrics)

    def _call_callback_hooks(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        for callback in self.callbacks:
            fn = getattr(callback, hook_name)
            fn(self, self.vessel, *args, **kwargs)

    def _min_loglevel(self):
        if not self.loggers:
            return LogLevel.PERIODIC
        else:
            # To save bandwidth
            return min(lg.loglevel for lg in self.loggers)


@contextmanager
def _wrap_context(obj):
    """Make any object a (possibly dummy) context manager."""

    if isinstance(obj, AbstractContextManager):
        # obj has __enter__ and __exit__
        with obj as ctx:
            yield ctx
    else:
        yield obj


def _named_collection(seq: Sequence[T]) -> Dict[str, T]:
    """Convert a list into a dict, where each item is named with its type."""
    res = {}
    retry_cnt: collections.Counter = collections.Counter()
    for item in seq:
        typename = type(item).__name__.lower()
        key = typename if retry_cnt[typename] == 0 else f"{typename}{retry_cnt[typename]}"
        retry_cnt[typename] += 1
        res[key] = item
    return res
