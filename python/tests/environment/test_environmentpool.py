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


def _make_pool(environments: list[Any]) -> EnvironmentPool:
    pool = EnvironmentPool.__new__(EnvironmentPool)
    pool.environments = environments
    pool.logger = logging.getLogger(__name__)
    pool._step_executor = None
    return pool


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
