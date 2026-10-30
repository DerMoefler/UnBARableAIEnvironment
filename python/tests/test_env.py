from __future__ import annotations

import math
from numbers import Real
from pprint import pprint
from typing import List, Optional

import pytest

import bar_ai
from src.environment.bar_environment import BAR_Environment


EPISODES = 5
MAX_STEPS_PER_EPISODE = 10_000


def create_actions(
    unit_ids: List[int],
    target_unit_id: Optional[int],
) -> List[bar_ai.Action]:
    """Erstellt gültige Testaktionen für alle lebenden eigenen Units."""
    actions: List[bar_ai.Action] = []

    for unit_id in unit_ids:
        action = bar_ai.Action()
        action.unit_id = unit_id
        action.team_id = 0
        action.ally_team_id = 0

        if target_unit_id is None:
            action.action_id = bar_ai.ActionId.MoveRight
            action.target_unit_id = 0
        else:
            action.action_id = bar_ai.ActionId.Attack
            action.target_unit_id = target_unit_id

        actions.append(action)

    return actions


@pytest.fixture
def env():
    """
    Erstellt eine Environment-Instanz für den vollständigen
    Mehr-Episoden-Test.
    """
    environment = BAR_Environment(training_team_id = 0)

    try:
        yield environment
    finally:
        environment.close()


def test_multiple_episodes_run_to_natural_completion(env):
    """
    Prüft, dass mehrere Episoden mit derselben Environment-Instanz
    vollständig und unabhängig voneinander durchlaufen.

    Der Test schlägt insbesondere fehl, wenn:

    - reset() den Episodenzustand nicht zurücksetzt,
    - eine Episode keinen einzigen Step ausführt,
    - eine Episode das Step-Limit überschreitet,
    - terminated und truncated gleichzeitig True sind,
    - eine Episode technisch abgebrochen wird,
    - kein fachliches EpisodeOutcome vorhanden ist,
    - info und Rückgabewerte widersprüchlich sind,
    - Observation oder Reward während der Episode ungültig sind,
    - nach einer Episode keine weitere Episode gestartet werden kann.
    """
    environment_id = id(env)
    completed_episodes = []

    for episode_index in range(EPISODES):
        episode_number = episode_index + 1

        print(
            f"\n=== EPISODE "
            f"{episode_number}/{EPISODES}: RESET ==="
        )

        try:
            obs, reset_info = env.reset()
        except Exception as exc:
            pytest.fail(
                f"Episode {episode_number}: reset() ist "
                f"fehlgeschlagen: {type(exc).__name__}: {exc}"
            )

        # Es muss über alle Episoden dieselbe Instanz verwendet werden.
        assert id(env) == environment_id, (
            f"Episode {episode_number}: Die Environment-Instanz "
            "wurde unerwartet ausgetauscht."
        )

        assert obs is not None, (
            f"Episode {episode_number}: reset() hat keine "
            "Observation zurückgegeben."
        )

        assert isinstance(obs, dict), (
            f"Episode {episode_number}: Die Observation aus reset() "
            "muss ein Dictionary sein, erhalten wurde "
            f"{type(obs).__name__}."
        )

        assert obs, (
            f"Episode {episode_number}: Die Observation aus reset() "
            "ist leer."
        )

        assert isinstance(reset_info, dict), (
            f"Episode {episode_number}: reset() muss info als "
            "Dictionary zurückgeben, erhalten wurde "
            f"{type(reset_info).__name__}."
        )

        assert reset_info.get("episode_step") == 0, (
            f"Episode {episode_number}: episode_step wurde durch "
            "reset() nicht auf 0 zurückgesetzt. Erhalten wurde "
            f"{reset_info.get('episode_step')!r}."
        )

        # Prüft auch die internen Episode-Flags auf alten Zustand.
        assert env.terminated is False, (
            f"Episode {episode_number}: terminated ist direkt nach "
            "reset() noch True. Vermutlich wurde der Zustand der "
            "vorherigen Episode nicht zurückgesetzt."
        )

        assert env.truncated is False, (
            f"Episode {episode_number}: truncated ist direkt nach "
            "reset() noch True. Vermutlich wurde der Zustand der "
            "vorherigen Episode nicht zurückgesetzt."
        )

        print("environment id:", id(env))
        print("n_obs:", len(obs))
        print("reset info:")
        pprint(reset_info, sort_dicts=False)

        terminated = False
        truncated = False
        step_count = 0
        reward: Real = 0
        info = reset_info

        while not (terminated or truncated):
            assert step_count < MAX_STEPS_PER_EPISODE, (
                f"Episode {episode_number} wurde nach "
                f"{MAX_STEPS_PER_EPISODE} Schritten weder natürlich "
                "beendet noch technisch abgeschnitten."
            )

            assert isinstance(obs, dict), (
                f"Episode {episode_number}, Schritt {step_count}: "
                "Observation muss ein Dictionary sein."
            )

            assert obs, (
                f"Episode {episode_number}, Schritt {step_count}: "
                "Observation ist während der laufenden Episode leer."
            )

            try:
                units = [
                    agent_array.item()
                    for agent_array in obs.values()
                ]
            except Exception as exc:
                pytest.fail(
                    f"Episode {episode_number}, Schritt {step_count}: "
                    "Die Observation konnte nicht in UnitData-Objekte "
                    f"umgewandelt werden: {type(exc).__name__}: {exc}"
                )

            own_alive_unit_ids = [
                int(unit.unit_id)
                for unit in units
                if int(unit.team_id) == 0
                and not bool(unit.is_dead)
                and float(unit.health) > 0.0
            ]

            enemy_alive_units = [
                unit
                for unit in units
                if int(unit.team_id) != 0
                and not bool(unit.is_dead)
                and float(unit.health) > 0.0
            ]

            target_unit_id = (
                int(enemy_alive_units[0].unit_id)
                if enemy_alive_units
                else None
            )

            actions = create_actions(
                own_alive_unit_ids,
                target_unit_id,
            )

            try:
                result = env.step(actions)
            except Exception as exc:
                pytest.fail(
                    f"Episode {episode_number}, "
                    f"Schritt {step_count + 1}: env.step() ist "
                    f"fehlgeschlagen: {type(exc).__name__}: {exc}"
                )

            assert isinstance(result, tuple), (
                f"Episode {episode_number}, Schritt {step_count + 1}: "
                "env.step() muss ein Tupel zurückgeben, erhalten "
                f"wurde {type(result).__name__}."
            )

            assert len(result) == 5, (
                f"Episode {episode_number}, Schritt {step_count + 1}: "
                "env.step() muss genau fünf Werte zurückgeben, "
                f"erhalten wurden {len(result)}."
            )

            obs, reward, terminated, truncated, info = result
            step_count += 1

            assert isinstance(terminated, bool), (
                f"Episode {episode_number}, Schritt {step_count}: "
                "terminated muss bool sein, erhalten wurde "
                f"{type(terminated).__name__}."
            )

            assert isinstance(truncated, bool), (
                f"Episode {episode_number}, Schritt {step_count}: "
                "truncated muss bool sein, erhalten wurde "
                f"{type(truncated).__name__}."
            )

            assert not (terminated and truncated), (
                f"Episode {episode_number}, Schritt {step_count}: "
                "terminated und truncated dürfen nicht gleichzeitig "
                "True sein."
            )

            assert isinstance(reward, Real) and not isinstance(
                reward,
                bool,
            ), (
                f"Episode {episode_number}, Schritt {step_count}: "
                "Reward muss numerisch sein, erhalten wurde "
                f"{type(reward).__name__}."
            )

            assert math.isfinite(float(reward)), (
                f"Episode {episode_number}, Schritt {step_count}: "
                f"Reward ist nicht endlich: {reward!r}."
            )

            assert isinstance(info, dict), (
                f"Episode {episode_number}, Schritt {step_count}: "
                "info muss ein Dictionary sein, erhalten wurde "
                f"{type(info).__name__}."
            )

            assert info.get("episode_step") == step_count, (
                f"Episode {episode_number}, Schritt {step_count}: "
                "info['episode_step'] stimmt nicht mit der Anzahl "
                "ausgeführter Steps überein. Erhalten wurde "
                f"{info.get('episode_step')!r}."
            )

            assert info.get("terminated") is terminated, (
                f"Episode {episode_number}, Schritt {step_count}: "
                "Der terminated-Wert in info widerspricht dem "
                "Rückgabewert."
            )

            assert info.get("truncated") is truncated, (
                f"Episode {episode_number}, Schritt {step_count}: "
                "Der truncated-Wert in info widerspricht dem "
                "Rückgabewert."
            )

            assert info.get("reward") == reward, (
                f"Episode {episode_number}, Schritt {step_count}: "
                "Der Reward in info widerspricht dem Rückgabewert."
            )

            if not (terminated or truncated):
                assert obs is not None, (
                    f"Episode {episode_number}, Schritt {step_count}: "
                    "Observation ist während der laufenden Episode "
                    "None."
                )

                assert isinstance(obs, dict) and obs, (
                    f"Episode {episode_number}, Schritt {step_count}: "
                    "Observation ist während der laufenden Episode "
                    "kein nicht-leeres Dictionary."
                )

                assert info.get("episode_outcome") is None, (
                    f"Episode {episode_number}, Schritt {step_count}: "
                    "Während die Episode läuft, darf noch kein "
                    "episode_outcome gesetzt sein. Erhalten wurde "
                    f"{info.get('episode_outcome')!r}."
                )

        print(
            f"\n=== EPISODE "
            f"{episode_number}/{EPISODES}: BEENDET ==="
        )
        print("environment id:", id(env))
        print("Schritte:", step_count)
        print("terminated:", terminated)
        print("truncated:", truncated)
        print("final reward:", reward)
        print("final info:")
        pprint(info, sort_dicts=False)

        # Eine Episode darf nicht direkt beim Reset bzw. ohne Step enden.
        assert step_count > 0, (
            f"Episode {episode_number} wurde beendet, ohne dass "
            "ein Step ausgeführt wurde."
        )

        # Dieser Integrationstest verlangt ein fachliches Spielende.
        # Technische Fehler oder Zeitlimits gelten als Fehlschlag.
        assert terminated is True, (
            f"Episode {episode_number} wurde nicht natürlich beendet."
        )

        assert truncated is False, (
            f"Episode {episode_number} wurde nach {step_count} "
            "Schritten abgeschnitten. Ein technischer Fehler, ein "
            "unbehandelter EngineStatus oder ein Episode-Limit darf "
            "in diesem Test nicht als erfolgreiche Episode gelten."
        )

        assert info.get("episode_outcome") is not None, (
            f"Episode {episode_number} wurde als terminated markiert, "
            "aber info enthält kein episode_outcome."
        )

        assert info.get("terminated") is True
        assert info.get("truncated") is False
        assert info.get("episode_step") == step_count

        completed_episodes.append(
            {
                "episode": episode_number,
                "steps": step_count,
                "outcome": info["episode_outcome"],
                "reward": reward,
            }
        )

    # Der Test ist erst erfolgreich, wenn wirklich alle Episoden
    # vollständig durchlaufen wurden.
    assert len(completed_episodes) == EPISODES, (
        f"Es wurden nur {len(completed_episodes)} von "
        f"{EPISODES} Episoden vollständig durchlaufen."
    )

    assert all(
        episode["steps"] > 0
        for episode in completed_episodes
    )

    assert all(
        episode["outcome"] is not None
        for episode in completed_episodes
    )

    print("\n=== ZUSAMMENFASSUNG ===")
    pprint(completed_episodes, sort_dicts=False)