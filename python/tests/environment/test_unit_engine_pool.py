#!/usr/bin/env python3

import pytest
import threading

from concurrent.futures import Future
from unittest.mock import MagicMock

import bar_ai
from src.environment.engine_pool import EngineSessionPool, SessionState

def wait_for_maintenance(pool: EngineSessionPool, timeout: float = 2.0) -> None:
    """
    Helper to wait for all scheduled operations to finish.
    """
    while True:
        with pool._lock:
            futures: tuple[Future[None], ...] = tuple(pool._maintenance_futures)

        if not futures:
            return
        for future in futures:
            future.result(timeout=timeout)

@pytest.fixture
def sessions() -> dict[int, MagicMock]:
    return {}

@pytest.fixture
def pool(sessions: dict[int, MagicMock]) -> EngineSessionPool:
    """
    Provide one pool with two successfully starting mock sessions.
    """
    def session_factory(slot_id: int) -> MagicMock:
        # --- Mock session ---
        session = MagicMock(name=f"engine_session_{slot_id}")
        session.get_engine_id.return_value = slot_id
        session.is_running.return_value = True
        session.wait_for_engine_status.return_value = True

        # --- Mock shared memory for team 0 ---
        shared_memory = MagicMock(name=f"shared_memory_{slot_id}_team_0")
        shared_memory.get_engine_status.return_value = (bar_ai.EngineStatus.READY)
        session.shared_memories = {0: shared_memory,}

        session.get_shared_memory.side_effect = (
            lambda team_id: session.shared_memories[
                team_id
            ]
        )


        sessions[slot_id] = session
        return session

    engine_pool = EngineSessionPool(pool_size=2,session_factory=session_factory, unbarable_ai_team_ids=(0,))

    yield engine_pool

    engine_pool.close()

def test_invalid_pool_size_is_rejected() -> None:
    session_factory = MagicMock()

    with pytest.raises(ValueError, match="Pool size must be greater than zero"):
        invalid_pool = EngineSessionPool(pool_size=0, session_factory=session_factory, unbarable_ai_team_ids=(0,))

def test_pool_is_initially_not_running(pool: EngineSessionPool) -> None:
    assert pool.is_running() is False

def test_start_publishes_all_sessions(pool: EngineSessionPool, sessions: dict[int, MagicMock]) -> None:
    pool.start()
    wait_for_maintenance(pool)

    assert pool.is_running() is True
    assert set(sessions) == {0, 1}
    assert set(pool._engine_slots) == {0, 1}

    for slot_id, session in sessions.items():
        session.start.assert_called_once_with()
        session.is_running.assert_called_once_with()

        assert pool._engine_slots[slot_id].session is session
        assert pool._engine_slots[slot_id].state is SessionState.AVAILABLE

    available_slot_ids = {
        pool._available_engines.get_nowait()
        for _ in range(2)
    }

    assert available_slot_ids == {0, 1}
    assert pool._available_engines.empty()

def test_session_in_state_starting() -> None:
    """
    Tests that the engine is in SessionState.STARTING while the engine is starting.
    """
    start_entered = threading.Event()
    start_finished = threading.Event()

    session = MagicMock(name="engine_session")
    session.is_running.return_value = True

    def blocking_start() -> None:
        start_entered.set()
        if not start_finished.wait(timeout=2.0):
            raise TimeoutError("Test did not release the startup operation")

    session.start.side_effect = blocking_start

    engine_pool = EngineSessionPool(pool_size=1,session_factory=lambda slot_id: session, unbarable_ai_team_ids=(0,))

    try:
        engine_pool.start()

        assert start_entered.wait(timeout=2.0)

        with engine_pool._lock:
            assert engine_pool._engine_slots[0].state is SessionState.STARTING

        assert engine_pool._available_engines.empty()
        start_finished.set()

        wait_for_maintenance(engine_pool)

        assert engine_pool._available_engines.get_nowait() == 0
        assert engine_pool._available_engines.empty()

    finally:
        start_finished.set()
        engine_pool.close()

def test_acquire_leases_available_session(pool: EngineSessionPool, sessions: dict[int, MagicMock]) -> None:
    pool.start()
    wait_for_maintenance(pool)

    acquired = pool.acquire(timeout=0.1)
    slot_id = acquired.get_engine_id()

    assert acquired is sessions[slot_id]
    assert pool._engine_slots[slot_id].state is SessionState.LEASED
    assert pool._available_engines.qsize() == 1

def test_acquire_times_out_when_all_sessions_are_leased(pool: EngineSessionPool) -> None:
    pool.start()
    wait_for_maintenance(pool)

    first = pool.acquire(timeout=0.1)
    second = pool.acquire(timeout=0.1)

    with pytest.raises(TimeoutError, match="No engine session became available"):
        pool.acquire(timeout=0.1)

def test_release_recycles_and_republishes_session(pool: EngineSessionPool) -> None:
    pool.start()
    wait_for_maintenance(pool)

    first = pool.acquire(timeout=0.1)
    second = pool.acquire(timeout=0.1)

    released_slot_id = first.get_engine_id()

    pool.release(first)
    wait_for_maintenance(pool)

    first.wait_for_engine_status.assert_called_with(bar_ai.EngineStatus.READY, timeout=20.0)

    assert pool._engine_slots[released_slot_id].state is SessionState.AVAILABLE

    acquired_again = pool.acquire(timeout=0.1)

    assert acquired_again is first
    assert acquired_again is not second
    assert pool._engine_slots[released_slot_id].state is SessionState.LEASED

def test_released_session_is_recycling_while_waiting_for_ready(pool: EngineSessionPool) -> None:
    pool.start()
    wait_for_maintenance(pool)

    session = pool.acquire(timeout=0.1)
    # Also acquire second so we can test that there is no engine available while the first is recycling
    _ = pool.acquire(timeout=0.1)
    slot_id = session.get_engine_id()

    wait_entered = threading.Event()
    allow_ready = threading.Event()

    def blocking_wait(status: bar_ai.EngineStatus, timeout: float) -> bool:
        wait_entered.set()
        if not allow_ready.wait(timeout=2.0):
            raise TimeoutError("Test did not unblock recycling")
        return True

    session.wait_for_engine_status.side_effect = blocking_wait

    try:
        pool.release(session)
        assert wait_entered.wait(timeout=2.0)
        with pool._lock:
            assert pool._engine_slots[slot_id].state is SessionState.RECYCLING
        # Must be empty because we acquired the second and the first is now busy recyling
        assert pool._available_engines.empty()

        allow_ready.set()
        wait_for_maintenance(pool)

        with pool._lock:
            assert pool._engine_slots[slot_id].state is SessionState.AVAILABLE

    finally:
        allow_ready.set()

def test_releasing_same_session_twice_is_rejected(pool: EngineSessionPool) -> None:
    pool.start()
    wait_for_maintenance(pool)

    session = pool.acquire(timeout=0.1)
    pool.release(session)

    with pytest.raises(RuntimeError, match="Cannot release session in state"):
        pool.release(session)

    wait_for_maintenance(pool)

def test_release_marks_session_failed_when_it_never_becomes_ready(pool: EngineSessionPool) -> None:
    pool.start()
    wait_for_maintenance(pool)

    session = pool.acquire(timeout=0.1)
    # Also acquire second engine so none is acquireable
    _ = pool.acquire(timeout=0.1)
    slot_id = session.get_engine_id()

    session.wait_for_engine_status.reset_mock()
    session.wait_for_engine_status.return_value = False

    pool.release(session)
    wait_for_maintenance(pool)

    assert pool._engine_slots[slot_id].state is SessionState.FAILED

    with pytest.raises(TimeoutError):
        pool.acquire(timeout=0.01)
