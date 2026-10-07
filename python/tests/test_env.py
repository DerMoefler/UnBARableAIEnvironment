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


@pytest.fixture
def env():
    """Erstellt die BAR-Umgebung und schließt sie nach dem Test."""
    environment = BAR_Environment()

    yield environment

    environment.close()


@pytest.mark.parametrize("episode", range(EPISODES))
def test_reset_and_step_for_multiple_episodes(env, episode):
    """
    Prüft den Reset- und Step-Vertrag über mehrere Episoden.

    Der Test schlägt fehl, wenn:
    - reset() keine Observation oder kein Info-Dictionary zurückgibt,
    - step() ungültige Rückgabewerte liefert,
    - eine Episode das festgelegte Schrittlimit überschreitet.
    """
    obs, info = env.reset()

    assert obs is not None, (
        f"Episode {episode}: reset() hat keine Observation zurückgegeben."
    )
    assert isinstance(info, dict), (
        f"Episode {episode}: reset() muss ein Dictionary als info zurückgeben, "
        f"erhalten wurde {type(info).__name__}."
    )

    print(f"\n=== EPISODE {episode + 1}/{EPISODES}: RESET ===")
    print("obs type:", type(obs).__name__)
    print("n_obs:", len(obs) if hasattr(obs, "__len__") else "unknown")
    print("info:")
    pprint(info, sort_dicts=False)

    terminated = False
    truncated = False
    step_count = 0

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
            f"env.step() muss 5 Werte zurückgeben, erhalten wurden {len(result)}."
        )

        obs, reward, terminated, truncated, info = result
        step_count += 1

        assert obs is not None, (
            f"Episode {episode + 1}, Schritt {step_count}: "
            "Observation ist None."
        )
        assert isinstance(reward, Real), (
            f"Episode {episode + 1}, Schritt {step_count}: "
            f"Reward muss numerisch sein, erhalten wurde "
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
    print("Schritte:", step_count)
    print("terminated:", terminated)
    print("truncated:", truncated)
    print("final reward:", reward)
    print("final info:")
    pprint(info, sort_dicts=False)

    assert terminated or truncated