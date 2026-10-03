from __future__ import annotations

from pprint import pprint

from src.environment.bar_environment import BAR_Environment

import bar_ai

def run_reset_step_test():
    """
    Exercises the environment reset and step contract until the episode ends.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If reset does not return observations and an info dictionary.
    """
    env = BAR_Environment()
    try:
        obs, info = env.reset()

        print("=== RESET TEST PASSED ===")
        print("obs type:", type(obs).__name__)
        print("n_obs:", len(obs) if hasattr(obs, "__len__") else "unknown")
        print("info:")
        pprint(info, sort_dicts=False)

        print(obs)

        assert obs is not None
        assert isinstance(info, dict)
        terminated = False
        truncated = False
        action = bar_ai.Action()

        action.unit_id = 1
        action.team_id = 0
        action.ally_team_id = 0
        action.action_id = bar_ai.ActionId.MoveUp
        action.target_unit_id = 0

        while not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(action)
            print("=== STEP TEST PASSED ===")
            print("obs type:", type(obs).__name__)
            print("reward:", reward)
            print("terminated:", terminated)
            print("truncated:", truncated)
            print("info:")
            pprint(info, sort_dicts=False)
        if terminated:
            print("Episode ended naturally (terminated).")
        elif truncated:
            print("Episode ended artificially (truncated).")

    finally:
        env.close()


if __name__ == "__main__":
    run_reset_step_test()