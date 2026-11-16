import numpy as np
import pytest

from src.environment.stepinfo import StepInfo


def test_update_stepinfo_accumulates_episode_metrics() -> None:
    stepinfo = StepInfo(action_dim=3)

    stepinfo.update_stepinfo(
        episode_number=1,
        action_types=[0, 1],
        reward=np.array([1.0, 3.0]),
        terminated=False,
        truncated=False,
        info={
            "own_alive_count": 2,
            "enemy_alive_count": 3,
            "reward_info": {
                "enemy_damage_done": 4.0,
                "own_damage_taken": 1.0,
            },
        },
        active_mask=np.array([[1.0], [0.0]]),
    )
    stepinfo.update_stepinfo(
        episode_number=1,
        action_types=[2, 1],
        reward=2.0,
        terminated=True,
        truncated=False,
        info={
            "episode_outcome": "won",
            "own_alive_count": 1,
            "enemy_alive_count": 0,
            "reward_info": {
                "enemy_damage_done": 3.0,
                "own_damage_taken": 2.0,
            },
        },
    )

    assert stepinfo.episode_reward == pytest.approx(4.0)
    assert stepinfo.step_count == 2
    assert stepinfo.done
    assert stepinfo.terminated
    assert not stepinfo.truncated
    assert stepinfo.action_counts.tolist() == [1, 1, 1]
    assert stepinfo.action_trace == [[0, 1], [2, 1]]
    assert stepinfo.episode_won
    assert not stepinfo.episode_lost
    assert stepinfo.final_ally_alive == 1
    assert stepinfo.final_enemy_alive == 0
    assert stepinfo.enemy_damage_done == pytest.approx(7.0)
    assert stepinfo.own_damage_taken == pytest.approx(3.0)
    assert stepinfo.last_info_keys == sorted(
        [
            "episode_outcome",
            "own_alive_count",
            "enemy_alive_count",
            "reward_info",
        ]
    )


def test_update_stepinfo_rejects_another_episode_after_done() -> None:
    stepinfo = StepInfo(action_dim=2)
    stepinfo.update_stepinfo(
        episode_number=1,
        action_types=[0],
        reward=1.0,
        terminated=True,
        truncated=False,
        info={},
    )

    with pytest.raises(RuntimeError, match="after the episode has ended"):
        stepinfo.update_stepinfo(
            episode_number=2,
            action_types=[1],
            reward=1.0,
            terminated=False,
            truncated=False,
            info={},
        )
