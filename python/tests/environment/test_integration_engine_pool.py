#!/usr/bin/env python3

import os
import pytest

from pathlib import Path

from src.environment.engine_session import EngineSession
from src.environment.engine_pool import EngineSessionPool, SessionState
from tests.src.helpers import wait_until
from tests.src.helpers import IntegrationScenario, load_base_config, create_scenario_config, write_config

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

    pool_size = 2

    scenario = IntegrationScenario(
        name="engine_pool_normal_3v3",
        number_of_engines=pool_size,
    )

    base_config = (
        load_base_config()
    )

    run_name = (
        "integration_engine_pool_normal_3v3"
    )

    config = create_scenario_config(
        base_config=base_config,
        scenario=scenario,
        run_name=run_name,
    )

    config_path = write_config(
        config=config,
        destination=(
            tmp_path
            / "engine_pool"
            / "training_config.json"
        ),
    )

    assert config_path.is_file()

    pool = EngineSessionPool.from_config(
        config
    )

    process_ids: set[int] = set()

    try:
        # --- Pool starts and creates actual processes ---
        pool.start()

        wait_until(
            lambda: (
                len(pool._engine_slots) == pool_size
                and all(
                    pool._engine_slots[slot_id].state
                    is SessionState.AVAILABLE
                    for slot_id in range(pool_size)
                )
            ),
            timeout=30.0,
        )

        sessions: dict[
            int,
            EngineSession,
        ] = {
            slot_id: slot.session
            for slot_id, slot
            in pool._engine_slots.items()
        }

        assert set(sessions) == set(
            range(pool_size)
        )

        for session in sessions.values():
            assert session.is_running()
            assert session.process_id is not None
            assert process_exists(session.process_id)

        process_ids = {
            session.process_id
            for session in sessions.values()
        }

        assert len(process_ids) == pool_size

        # --- Acquire actual processes ---
        leased_session = [
            pool.acquire(timeout=5.0) for _ in range(pool_size)
        ]
        assert len({id(session) for session in leased_session}) == pool_size

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
