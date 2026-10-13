#!/usr/bin/env python3

import os
import pytest

from pathlib import Path

from src.environment.engine_session import EngineSession, EngineSessionConfig
from src.environment.engine_pool import EngineSessionPool, SessionState
from tests.src.helpers import wait_until

def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    return True

@pytest.mark.integration
def test_pool_starts_publishes_and_closes_real_engines(tmp_path: Path) -> None:
    created_sessions: dict[int, EngineSession] = {}

    def session_factory(slot_id: int) -> EngineSession:
        session = EngineSession(cfg=EngineSessionConfig(engine_id=slot_id))

        created_sessions[slot_id] = session
        return session

    pool = EngineSessionPool(pool_size=4, session_factory=session_factory)

    try:
        pool.start()

        wait_until(
            lambda: (
                len(created_sessions) == 4
                and all(
                    pool._engine_slots[0].state
                    is SessionState.AVAILABLE
                    for slot_id in range(4)
                )
            ),
            timeout=30.0
        )

        assert set(created_sessions) == {0, 1, 2, 3}

        for session in created_sessions.values():
            assert session.is_running()
            assert session.process_id is not None
            assert process_exists(session.process_id)

        process_ids = {
            session.process_id
            for session in created_sessions.values()
        }

        assert len(process_ids) == 4

    finally:
        pool.close()

        wait_until(
            lambda: all(
                not process_exists(process_id)
                for process_id in process_ids
            ),
            timeout=10.0,
        )
