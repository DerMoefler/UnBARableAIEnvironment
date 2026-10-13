#!/usr/bin/env python3

import pytest
import threading

from concurrent.futures import Future
from unittest.mock import MagicMock

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
        session = MagicMock(name=f"engine_session_{slot_id}")
        session.is_running.return_value = True
        sessions[slot_id] = session
        return session

    engine_pool = EngineSessionPool(pool_size=2,session_factory=session_factory)

    yield engine_pool

    engine_pool.close()

def test_invalid_pool_size_is_rejected() -> None:
    session_factory = MagicMock()

    with pytest.raises(ValueError, match="Pool size must be greater than zero"):
        invalid_pool = EngineSessionPool(pool_size=0, session_factory=session_factory)

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

    engine_pool = EngineSessionPool(pool_size=1,session_factory=lambda slot_id: session)

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
        engine_pool.close()
