from __future__ import annotations

from numbers import Real
from pprint import pprint

import pytest

import bar_ai
from src.environment.bar_environment import BAR_Environment


EPISODES = 5
MAX_STEPS_PER_EPISODE = 10_000


def create_move_up_action() -> bar_ai.Action:
    """Erstellt eine gültige MoveUp-Testaktion."""
    action = bar_ai.Action()
    action.unit_id = 1
    action.team_id = 0
    action.ally_team_id = 0
    action.action_id = bar_ai.ActionId.MoveUp
    action.target_unit_id = 0

    return action


@pytest.fixture(scope="module")
def env():
    """
    Erstellt genau eine BAR-Umgebung für alle Tests in diesem Modul.

    Das Environment wird nach dem letzten Test geschlossen.
    """
    environment = BAR_Environment()

    yield environment

    environment.close()


@pytest.mark.parametrize(
    "episode",
    range(EPISODES),
    ids=lambda episode: f"episode-{episode + 1}",
)
def test_reset_and_step_for_multiple_episodes(env, episode):
    """
    Prüft eine Episode mit einer gemeinsam verwendeten Environment-Instanz.

    Jede Episode erscheint als eigener Pytest-Testfall.
    Das Environment wird zwischen den Episoden mit reset() zurückgesetzt.
    """
    obs, info = env.reset()

    assert obs is not None, (
        f"Episode {episode + 1}: "
        "reset() hat keine Observation zurückgegeben."
    )
    assert isinstance(info, dict), (
        f"Episode {episode + 1}: "
        "reset() muss ein Dictionary als info zurückgeben, "
        f"erhalten wurde {type(info).__name__}."
    )

    print(f"\n=== EPISODE {episode + 1}/{EPISODES}: RESET ===")
    print("environment id:", id(env))
    print("obs type:", type(obs).__name__)
    print(
        "n_obs:",
        len(obs) if hasattr(obs, "__len__") else "unknown",
    )
    print("info:")
    pprint(info, sort_dicts=False)

    terminated = False
    truncated = False
    step_count = 0
    reward: Real = 0

    while not (terminated or truncated):
        assert step_count < MAX_STEPS_PER_EPISODE, (
            f"Episode {episode + 1} wurde nach "
            f"{MAX_STEPS_PER_EPISODE} Schritten weder beendet "
            "noch abgeschnitten."
        )

        action = create_move_up_action()
        result = env.step(action)

        assert isinstance(result, tuple), (
            f"Episode {episode + 1}, Schritt {step_count}: "
            "env.step() muss ein Tupel zurückgeben."
        )
        assert len(result) == 5, (
            f"Episode {episode + 1}, Schritt {step_count}: "
            "env.step() muss 5 Werte zurückgeben, "
            f"erhalten wurden {len(result)}."
        )

        obs, reward, terminated, truncated, info = result
        step_count += 1

        # Nach dem Game-End darf obs None sein. Während die Episode
        # noch läuft, muss eine Observation vorhanden sein.
        if not (terminated or truncated):
            assert obs is not None, (
                f"Episode {episode + 1}, Schritt {step_count}: "
                "Observation ist während der laufenden Episode None."
            )

        assert isinstance(reward, Real), (
            f"Episode {episode + 1}, Schritt {step_count}: "
            "Reward muss numerisch sein, erhalten wurde "
            f"{type(reward).__name__}."
        )
        assert isinstance(terminated, bool), (
            f"Episode {episode + 1}, Schritt {step_count}: "
            "terminated muss ein bool sein."
        )
        assert isinstance(truncated, bool), (
            f"Episode {episode + 1}, Schritt {step_count}: "
            "truncated muss ein bool sein."
        )
        assert isinstance(info, dict), (
            f"Episode {episode + 1}, Schritt {step_count}: "
            "info muss ein Dictionary sein."
        )

    print(f"\n=== EPISODE {episode + 1}/{EPISODES}: BEENDET ===")
    print("environment id:", id(env))
    print("Schritte:", step_count)
    print("terminated:", terminated)
    print("truncated:", truncated)
    print("final obs:", obs)
    print("final reward:", reward)
    print("final info:")
    pprint(info, sort_dicts=False)

    assert terminated or truncated