import logging
from posixpath import ismount
import queue
import threading

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

from src.environment.engine_session import EngineSession

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
    def __init__(self, pool_size: int, session_factory: Callable[[int], EngineSession]):
        self.pool_size: int = pool_size
        if self.pool_size <= 0:
            raise ValueError("Pool size must be greater than zero.")

        self._session_factory: Callable[[int], EngineSession] = session_factory

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

        engine_slot.state = SessionState.READY
        self._publish_slot(slot_id)


    def _get_slot(self, slot_id: int) -> PoolSlot:
        """
        Thread safe access to engine at slot_id
        """
        with self._lock:
            return self._engine_slots[slot_id]

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

