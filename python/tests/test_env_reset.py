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

    finally:
        env.close()


if __name__ == "__main__":
    run_reset_test()