import logging
import queue
import threading
import time

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Mapping, Any

import bar_ai
from src.environment.engine_session import EngineSession, EngineSessionFactory
from src.environment.startscript_builder import StartScriptConfig

class SessionState(Enum):
    """
    The state of an engine session within the pool.

    Only sessions in READY should be in the ready queue.
    """
    STARTING = auto()
    READY = auto()
    AVAILABLE = auto()
    LEASED = auto()
    RECYCLING = auto()
    FAILED = auto()
    STOPPED = auto()

@dataclass
class PoolSlot:
    """
    A slot within the pool, essentially used to associate a SessionState with an EngineSession.
    """
    session: EngineSession
    state: SessionState

class EngineSessionPool:
    def __init__(self, pool_size: int, session_factory: Callable[[int], EngineSession], unbarable_ai_team_ids: tuple[int, ...]):
        self.pool_size: int = pool_size
        if self.pool_size <= 0:
            raise ValueError("Pool size must be greater than zero.")

        self._session_factory: Callable[[int], EngineSession] = session_factory
        self._unbarable_ai_team_ids: tuple[int, ...] = unbarable_ai_team_ids

        self._is_running: bool = False
        # TODO Currently only possible to close once, after start won't work
        self._closing: bool = False
        self._lock: threading.RLock = threading.RLock()

        self._engine_slots: dict[int, PoolSlot] = {}

        self._available_engines: queue.Queue[int] = queue.Queue(maxsize=self.pool_size)

        self._maintenance_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=pool_size,
            thread_name_prefix="engine-maintenance",
        )
        self._maintenance_futures: set[Future[None]] = set()


    def start(self) -> None:
        """
        Create all EngineSessions objects and start them asynchronously.

        This method returns after scheduling startup, meaning it does not wait for the actual
        initialization (loading the startscript etc).
        """
        with self._lock:
            if self._is_running:
                raise RuntimeError("Pool is already running")
            if self._closing:
                raise RuntimeError("Pool is currently closing and cannot be started.")

            self._is_running = True

            for slot_id in range(self.pool_size):
                session: EngineSession = self._session_factory(slot_id)

                engine_slot = PoolSlot(
                    session=session,
                    state=SessionState.STOPPED
                )
                self._engine_slots[slot_id] = engine_slot

                self._submit_maintenance(self._start_slot, slot_id)

    def close(self) -> None:
        """
        Stop accepting maintenance operations and wait for exisitng operations to finish.

        Calling close multiple times is allowed.
        """
        with self._lock:
            if self._closing:
                return
            self._closing = True
            self._is_running = False

        self._maintenance_executor.shutdown(wait=True, cancel_futures=True)

        # Copy the slots so we do not hold the pool lock while stopping potentially
        # slow external processes.
        with self._lock:
            slots = tuple(self._engine_slots.items())

        for slot_id, slot in slots:
            try:
                slot.session.stop()
            except Exception:
                logging.exception(f"Failed to close Engine at slot {slot_id}")
            finally:
                with self._lock:
                    slot.state = SessionState.STOPPED

    def acquire(self, timeout: float | None) -> EngineSession:
        with self._lock:
            if not self._is_running:
                raise RuntimeError("Pool has not yet been started")
            if self._closing:
                raise RuntimeError("Pool is closing")

        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            queue_timeout = None
            if deadline is not None:
                queue_timeout = deadline - time.monotonic()

                if queue_timeout <= 0:
                    raise TimeoutError(f"No engine session became available while waiting for {timeout} seconds.")

            try:
                slot_id: int = self._available_engines.get(timeout=queue_timeout)
            except queue.Empty as exception:
                raise TimeoutError("No engine session became available")

            with self._lock:
                if self._closing:
                    raise RuntimeError("Pool is closing")
                slot: PoolSlot = self._engine_slots[slot_id]

                assert slot.state == SessionState.AVAILABLE
                assert slot.session.is_running()
                assert slot.session.shared_memory is not None
                assert slot.session.shared_memory.get_engine_status() == bar_ai.EngineStatus.READY

                slot.state = SessionState.LEASED
                logging.info(f"Engine slot {slot_id} has been acquired")

                return slot.session

    def release(self, session: EngineSession) -> None:
        """
        Releases a previously acquired engine.

        This method obviously requires the session to actually have been LEASED. If so,
        it will put the slot state to RECYCLING and only after that asynchronously call
        "_recycle_slot" which will wait for the engine to become bar_ai.EngineStatus.READY.
        This is done to avoid race conditions, because otherwise the engine could be reacquired
        without having been recycled.
        """
        with self._lock:
            if not self._is_running or self._closing:
                raise RuntimeError("Pool is not running")

            slot_id = session.get_engine_id()
            try:
                slot = self._engine_slots[slot_id]
            except KeyError:
                raise ValueError("Session does not belong to this pool") from None

            if slot.session is not session:
                raise ValueError(
                    "Session does not belong to this pool"
                    "or mismatch between EngineSession id and slot id"
                )

            if slot.state is not SessionState.LEASED:
                raise RuntimeError(f"Cannot release session in state {slot.state.name}")

            if not session.is_running():
                slot.state = SessionState.FAILED
                logging.error(f"Released session at slot {slot_id} is no longer running")
                return

            slot.state = SessionState.RECYCLING
            self._submit_maintenance(self._recycle_slot, slot_id)

    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    def _start_slot(self, slot_id: int) -> None:
        """
        Maintenance-thread operator for starting the EngineSession at slot_id.

        Requires the engine to be in state STOPPED.
        """
        engine_slot = self._get_slot(slot_id)

        if engine_slot.state is not SessionState.STOPPED:
            raise RuntimeError(f"Engine at slot {slot_id} attempted to start while in state {engine_slot.state.name}")

        try:
            engine_slot.state = SessionState.STARTING
            _ = engine_slot.session.start()
        except BaseException as exception:
            engine_slot.state = SessionState.FAILED
            logging.exception(f"Engine at slot {slot_id} failed during startup.", exc_info=exception)
            return


        self._wait_for_ready(slot_id, timeout=20.0)

        if engine_slot.state is SessionState.FAILED:
            return

        self._publish_slot(slot_id)

    def _recycle_slot(self, slot_id: int) -> None:
        """
        Maintenance-thread operator for recycling an EngineSession at slot_id.

        Requires the engine to be in state RECYCLING before calling to avoid race conditions.
        """
        with self._lock:
            engine_slot = self._get_slot(slot_id)

            if engine_slot.state != SessionState.RECYCLING:
                raise RuntimeError(
                    "Can only recycle engines already in state RECYLING. "
                    f"Actual state: {engine_slot.state.name}"
                )

        self._wait_for_ready(slot_id, timeout=20.0)

        with self._lock:
            if engine_slot.state == SessionState.FAILED:
                return

        self._publish_slot(slot_id)


    def _get_slot(self, slot_id: int) -> PoolSlot:
        """
        Thread safe access to engine at slot_id
        """
        with self._lock:
            return self._engine_slots[slot_id]

    def _wait_for_ready(self, slot_id: int, timeout: float) -> None:
        """
        Maintenance-thread operator for waiting until the EngineSession is ready.

        Requires the engine to be either in STARTING or RECYCLING.
        """
        engine_slot = self._get_slot(slot_id)

        if engine_slot.state not in {SessionState.STARTING, SessionState.RECYCLING}:
            raise RuntimeError("Can only switch to ready from STARTING or RECYCLING state. "
                               f"Actual state: {engine_slot.state.name}")

        success: bool = engine_slot.session.wait_for_engine_status(
            bar_ai.EngineStatus.READY,
            timeout=timeout
        )

        if not success:
            logging.error(f"Engine at slot {slot_id} did not become ready in {timeout} seconds")
            engine_slot.state = SessionState.FAILED
            return

        engine_slot.state = SessionState.READY

    def _publish_slot(self, slot_id: int) -> None:
        """
        Put the Engine at slot_id into the Queue, making it available to acquire().

        This is the only method putting entries into the Queue.
        """
        with self._lock:
            engine_slot = self._get_slot(slot_id)

            if engine_slot.state is not SessionState.READY:
                raise RuntimeError(f"Engine at slot {slot_id} cannot be published, as it is not in {SessionState.READY.name}")
            if not engine_slot.session.is_running():
                raise RuntimeError(f"Engine at slot {slot_id} cannot be published, as it is not running.")

            engine_slot.state = SessionState.AVAILABLE
            # State changed before putting into queue as another thread might immediately try acquiring the slot.
            self._available_engines.put_nowait(slot_id)

        logging.info(f"Engine at slot {slot_id} available.")


    def _submit_maintenance(self, function: Callable[[int], None], slot_id: int) -> None :
        """
        Helper to execute functions using the ThreadPoolExecutor, i.e. schedule asynchronous calls.
        """
        future = self._maintenance_executor.submit(function, slot_id)
        with self._lock:
            self._maintenance_futures.add(future)

        def maintenance_completed(future: Future[None]) -> None:
            """
            Helper function used as a callback once the operation has finished.

            Simply discards the future, i.e. removes it from the set of _maintenance_futures
            and catches unhandeled errors.
            """
            with self._lock:
                self._maintenance_futures.discard(future)
            try:
                future.result()
            except BaseException:
                logging.exception(f"Unhandeled maintenaced error for Engine at slot {slot_id}.")

        future.add_done_callback(maintenance_completed)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> "EngineSessionPool":
        startscript_config = (
            StartScriptConfig.from_config(config)
        )

        if startscript_config.self_play:
            unbarable_ai_team_ids = (
                startscript_config.training_team_id,
                startscript_config.opponent_team_id,
            )
        else:
            unbarable_ai_team_ids = (
                startscript_config.training_team_id,
            )

        session_factory = EngineSessionFactory(
            startscript_config=startscript_config,
            unbarable_ai_team_ids=unbarable_ai_team_ids,
        )

        return cls(
            pool_size=config["number_of_engines"],
            session_factory=session_factory,
            unbarable_ai_team_ids=unbarable_ai_team_ids
        )

