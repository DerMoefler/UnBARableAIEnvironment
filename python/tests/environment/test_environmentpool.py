from __future__ import annotations

import logging
import threading
from typing import Any

import pytest

from src.environment.environmentpool import EnvironmentPool


class BarrierEnvironment:
    def __init__(
        self,
        environment_id: int,
        barrier: threading.Barrier,
    ) -> None:
        self.environment_id = environment_id
        self.barrier = barrier

    def step(self, actions: list[Any]) -> tuple[int, list[Any]]:
        self.barrier.wait(timeout=2.0)
        return self.environment_id, actions

    def close(self) -> None:
        pass


class ResetEnvironment:
    def __init__(self, observation: Any) -> None:
        self.observation = observation
        self.reset_count = 0

    def reset(self) -> Any:
        self.reset_count += 1
        return self.observation


def _make_pool(environments: list[Any]) -> EnvironmentPool:
    pool = EnvironmentPool.__new__(EnvironmentPool)
    pool.environments = environments
    pool.logger = logging.getLogger(__name__)
    pool._step_executor = None
    return pool


def test_reset_environment_resets_only_the_requested_environment() -> None:
    environments = [
        ResetEnvironment({"env": 0}),
        ResetEnvironment({"env": 1}),
    ]
    pool = _make_pool(environments)

    observation = pool.reset_environment(1)

    assert observation == {"env": 1}
    assert [environment.reset_count for environment in environments] == [0, 1]


def test_reset_environment_rejects_out_of_range_id() -> None:
    pool = _make_pool([ResetEnvironment({"env": 0})])

    with pytest.raises(IndexError, match="Environment ID 1 is out of range"):
        pool.reset_environment(1)


def test_step_environments_steps_concurrently_and_keeps_order() -> None:
    barrier = threading.Barrier(2)
    pool = _make_pool(
        [
            BarrierEnvironment(0, barrier),
            BarrierEnvironment(1, barrier),
        ]
    )

    try:
        results = pool.step_environments(
            [["env-0-action"], ["env-1-action"]]
        )

        assert results == [
            (0, ["env-0-action"]),
            (1, ["env-1-action"]),
        ]
    finally:
        pool.close_environements()


def test_step_environments_requires_actions_for_every_environment() -> None:
    pool = _make_pool([BarrierEnvironment(0, threading.Barrier(1))])

    with pytest.raises(ValueError, match="one action list per environment"):
        pool.step_environments([])

    pool.close_environements()
