from enum import Enum
import bar_ai

class EpisodeOutcome(Enum):
    WON = "won"
    LOST = "lost"

RUNNING_STATUSES = {
    bar_ai.EngineStatus.STARTING,
    bar_ai.EngineStatus.READY,
    bar_ai.EngineStatus.RUNNING,
}

TERMINATED_STATUSES = {
    bar_ai.EngineStatus.GAME_WON,
    bar_ai.EngineStatus.GAME_LOST,
}

# TODO currently simply none (I do not believe any were
# written previously though tbh)
TRUNCATED_STATUSES = {
}
