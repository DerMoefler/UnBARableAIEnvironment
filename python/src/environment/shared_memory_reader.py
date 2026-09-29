from dataclasses import dataclass
import bar_ai


@dataclass(frozen=True)
class BARUnitView:
    unit_id: int
    unit_def_id: int
    unit_def_name: str
    human_name: str
    team_id: int
    ally_team_id: int
    health: float
    max_health: float
    pos_x: float
    pos_y: float
    pos_z: float
    los_radius: float
    air_los_radius: float
    is_dead: bool
    being_built: bool
    build_progress: float
    capture_progress: float
    paralyze_damage: float

    @property
    def pos(self):
        return (self.pos_x, self.pos_y, self.pos_z)

class SharedMemoryReader:
    shared_memory = bar_ai.
    def __init__(self, shared_memory=None):
        self.shared_memory = shared_memory

    def read_all(self):
        self.shared_memory.

