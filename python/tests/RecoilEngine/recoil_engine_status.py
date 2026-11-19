#!/usr/bin/env python3

import time
import pytest

from src.environment.engine_session import EngineSession, EngineSessionFactory
from src.environment.startscript_builder import StartScriptConfig

import bar_ai

# The UnBARableAI inside the RecoilEngine has a static constexpr member defining the
# regular timeout. The first event update is the only one that can (and also must be
# able to) exceed this limit, which is what is being tested here. Therefore, these
# timeouts must be kept in sync.
ENGINE_SIDE_GRPC_CONNCETION_TIMEOUT: float = 30.0

def wait_for_status(
    session: EngineSession,
    expected_status: bar_ai.EngineStatus,
    *,
    timeout: float,
    poll_interval: float = 0.05,
) -> None:
    assert session.shared_memory is not None, "Shared-memory has not been initialized"

    deadline = time.monotonic() + timeout

    last_status = session.shared_memory.get_engine_status()

    while time.monotonic() < deadline:
        assert session.is_running(), (
            f"Engine excisted while waiting for {expected_status}; "
            f"exit code: {session.exit_code()}"
        )
        last_status = session.shared_memory.get_engine_status()

        if last_status == expected_status:
            return

        time.sleep(poll_interval)

    pytest.fail(
        f"Engine did not read {expected_status} within {timeout} seconds. "
        f"Last status was {last_status}."
    )

@pytest.fixture
def session():
    """
    Creates an engine session instance.
    """
    startscript_config = StartScriptConfig()
    factory = EngineSessionFactory(startscript_config)
    session: EngineSession = factory(engine_id=100)

    try:
        yield session
    finally:
        session.stop()

@pytest.mark.timeout(90)
@pytest.mark.integration
def test_engine_session_does_not_timeout(session: EngineSession):
    session.start()

    assert session.is_running(), "Session did not start"
    assert session.shared_memory is not None, "Shared-memory has not been initialized"

    assert session.shared_memory.get_engine_status() == bar_ai.EngineStatus.STARTING, (
        "Wrong state after starting"
    )

    # Wait for first handle event update
    first_handle_event_update_timeout = 20
    wait_status, update_id = session.grpc_server.wait_for_next_update(
        previous_count=0,
        timeout=first_handle_event_update_timeout,
    )
    if wait_status == "timeout":
        pytest.fail(
            f"No handle event update after {first_handle_event_update_timeout} seconds."
        )
    if wait_status == "stopped":
        pytest.fail("gRPC server stopped unexpectedly.")
    # Assert handle event update worked as expected
    assert wait_status == "update"
    assert update_id is not None
    assert update_id == 1, "This is not the first update, first update id should be 1"
    assert session.shared_memory.get_engine_status() == bar_ai.EngineStatus.READY, (
        "Engine not READY on first handle event update"
    )

    # Wait longer than the normal gRPC connection timeout.
    time.sleep(ENGINE_SIDE_GRPC_CONNCETION_TIMEOUT + 5.0)

    # Assert that after the regular timeout, everything works as expected
    assert session.is_running(), "Engine exited while waiting for acknowledgement"
    assert session.exit_code() is None, (
        f"Engine existed unexpectedly with code {session.exit_code()}"
    )
    assert session.shared_memory.get_engine_status() == bar_ai.EngineStatus.READY, (
        "Engine left READY state before receiving an acknowledgement"
    )

    # Free update now
    session.grpc_server.ack_update(update_id=1)
    # After acknowledging the update, the status should basically instantaneously
    # change to RUNNING
    wait_for_status(
        session,
        bar_ai.EngineStatus.RUNNING,
        timeout=1.0,
    )

    assert session.is_running(), "Engine exited after acknowledgement"
