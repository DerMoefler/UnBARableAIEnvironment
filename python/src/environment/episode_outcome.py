from enum import Enum
import bar_ai

class EpisodeOutcome(Enum):
    WON = "won"
    LOST = "lost"

TERMINATED_STATUSES = {
    bar_ai.EngineStatus.READY,
    bar_ai.EngineStatus.RUNNING,
    bar_ai.EngineStatus.GAME_WON,
    bar_ai.EngineStatus.GAME_LOST,
}

TRUNCATED_STATUSES = {
    # TODO check if this is correct and nice lol
    bar_ai.EngineStatus.STARTING,
}
