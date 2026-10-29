from enum import Enum
import bar_ai

class EpisodeOutcome(Enum):
    WON = "won"
    LOST = "lost"

TERMINATED_STATUSES = {
    bar_ai.EngineStatus.GAME_ENDED,
    bar_ai.EngineStatus.TEAM_DIED,
    bar_ai.EngineStatus.AI_KILLED,
}

TRUNCATED_STATUSES = {
    bar_ai.EngineStatus.UNSPECIFIED_ERROR,
    bar_ai.EngineStatus.AI_CRASHED,
    bar_ai.EngineStatus.AI_FAILED_TO_INIT,
    bar_ai.EngineStatus.CONNECTION_LOST,
    bar_ai.EngineStatus.OTHER_REASON_ERROR,
}