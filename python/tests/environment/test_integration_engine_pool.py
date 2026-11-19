#!/usr/bin/env python3

import os
import pytest

from pathlib import Path

from src.environment.engine_session import EngineSession, EngineSessionConfig
from src.environment.engine_pool import EngineSessionPool, SessionState
from src.environment.startscript_builder import EngineMode, StartScriptConfig
from tests.src.helpers import wait_until

def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    return True

# Rather ugly monolothic test but I guess it gets the job done :)
@pytest.mark.integration
def test_pool_starts_publishes_acquires_releases_and_closes_real_engines(tmp_path: Path) -> None:
    created_sessions: dict[int, EngineSession] = {}

    def session_factory(slot_id: int) -> EngineSession:
        session = EngineSession(
            cfg=EngineSessionConfig(
                engine_id=slot_id,
                startscript_config=StartScriptConfig(
                    mode=EngineMode.TRAINING,
                    number_of_agents=3,
                ),
            )
        )

        created_sessions[slot_id] = session
        return session

    pool = EngineSessionPool(pool_size=4, session_factory=session_factory)

    process_ids: set[int] = set()

    try:
        # --- Pool starts and creates actual processes ---
        pool.start()

        wait_until(
            lambda: (
                len(created_sessions) == 4
                and all(
                    pool._engine_slots[slot_id].state
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

        # --- Acquire actual processes ---
        leased_session = [
            pool.acquire(timeout=5.0) for _ in range(4)
        ]
        assert len({id(session) for session in leased_session}) == 4

        for session in leased_session:
            slot_id = session.get_engine_id()
            assert pool._engine_slots[slot_id].state is SessionState.LEASED

        with pytest.raises(TimeoutError):
            pool.acquire(timeout=0.1)

        # --- Release and reacquired ---
        released_session = leased_session[0]
        released_slot_id = released_session.get_engine_id()

        pool.release(released_session)
        wait_until(lambda: pool._engine_slots[released_slot_id].state is SessionState.AVAILABLE, timeout = 30.0)

        reacquired_session = pool.acquire(timeout=5.0)
        assert reacquired_session is released_session
        assert pool._engine_slots[released_slot_id].state is SessionState.LEASED

    finally:
        pool.close()

        wait_until(
            lambda: all(
                not process_exists(process_id)
                for process_id in process_ids
            ),
            timeout=10.0,
        )
