from __future__ import annotations

from pprint import pprint

from src.environment.bar_environment import BAR_Environment


def run_reset_test():
    env = BAR_Environment()
    try:
        obs, info = env.reset()

        print("=== RESET TEST PASSED ===")
        print("obs type:", type(obs).__name__)
        print("n_obs:", len(obs) if hasattr(obs, "__len__") else "unknown")
        print("info:")
        pprint(info, sort_dicts=False)

        assert obs is not None
        assert isinstance(info, dict)
        for i in range(1):
            obs, reward, terminated, truncated, info = env.step(10) # not a real action
            print("=== STEP TEST PASSED ===")
            print("obs type:", type(obs).__name__)
            print("reward:", reward)
            print("terminated:", terminated)
            print("truncated:", truncated)
            print("info:")
            pprint(info, sort_dicts=False)

    finally:
        env.close()


if __name__ == "__main__":
    run_reset_test()