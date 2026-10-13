#!/usr/bin/env python3

# Collection of helpers / utility

import time

from collections.abc import Callable

def wait_until(condition: Callable[[], bool], timeout: float = 30.0, interval: float = 0.05) -> None:
    """
    Simple helper to sleep a thread, repeatedly checking if the condition has been satisfied yet.
    """
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if condition():
            return

        time.sleep(interval)

    raise TimeoutError("Condition was not satisfied before timeout")
