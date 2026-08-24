from dataclasses import dataclass
import time
from typing import Optional, Dict, Any, List, Mapping, Callable

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
    def __init__(
        self,
        ipc_factory: Optional[Callable[[], Any]] = None,
        ready_timeout_s: float = 10.0,
        poll_interval_s: float = 0.05,
    ) -> None:
        self._ipc_factory = ipc_factory
        self._ready_timeout_s = ready_timeout_s
        self._poll_interval_s = poll_interval_s
        self._ipc = None

    def connect(self) -> None:
        if self._ipc is not None:
            return
        if self._ipc_factory is None:
            raise RuntimeError(
                "No shared-memory IPC factory configured."
            )

        self._ipc = self._ipc_factory()
        deadline = time.monotonic() + self._ready_timeout_s
        last_exc = None
        while time.monotonic() < deadline:
            try:
                self._ipc.connect()
                return
            except Exception as exc:
                last_exc = exc
                time.sleep(self._poll_interval_s)

        self._ipc = None
        raise RuntimeError(
            "Failed to connect to BAR shared memory within "
            f"{self._ready_timeout_s:.2f}s"
        ) from last_exc

    def close(self) -> None:
        if self._ipc is None:
            return
        try:
            self._ipc.close()
        finally:
            self._ipc = None

    def _snapshot(self) -> Dict[str, Any]:
        self.connect()
        ipc = self._ipc
        if ipc is None:
            raise RuntimeError("Shared memory IPC is not connected.")
        snapshot = ipc.read_snapshot()
        if not snapshot:
            raise RuntimeError("Shared memory snapshot is empty or unavailable.")
        if "units_by_id" not in snapshot:
            raise KeyError("Shared memory snapshot missing 'units_by_id'.")
        return snapshot

    def get_world_frame(self) -> int:
        return int(self._snapshot().get("frame", -1))

    @staticmethod
    def _coerce_unit_view(rec: Mapping[str, Any]) -> BARUnitView:
        return BARUnitView(
            unit_id=int(rec["unit_id"]),
            unit_def_id=int(rec["unit_def_id"]),
            unit_def_name=str(rec.get("unit_def_name", "")),
            human_name=str(rec.get("human_name", "")),
            team_id=int(rec.get("team_id", -1)),
            ally_team_id=int(rec.get("ally_team_id", -1)),
            health=float(rec.get("health", 0.0)),
            max_health=float(rec.get("max_health", 0.0)),
            pos_x=float(rec.get("pos_x", 0.0)),
            pos_y=float(rec.get("pos_y", 0.0)),
            pos_z=float(rec.get("pos_z", 0.0)),
            los_radius=float(rec.get("los_radius", 0.0)),
            air_los_radius=float(rec.get("air_los_radius", 0.0)),
            is_dead=bool(rec.get("is_dead", False)),
            being_built=bool(rec.get("being_built", False)),
            build_progress=float(rec.get("build_progress", 1.0)),
            capture_progress=float(rec.get("capture_progress", 0.0)),
            paralyze_damage=float(rec.get("paralyze_damage", 0.0)),
        )

    def get_all_units(self) -> Dict[int, BARUnitView]:
        snap = self._snapshot()
        units = snap["units_by_id"]
        return {
            int(unit_id): self._coerce_unit_view(rec)
            for unit_id, rec in units.items()
        }
    
    def get_unit_by_id(self, unit_id: int) -> Optional[BARUnitView]:
        snap = self._snapshot()
        rec = snap["units_by_id"].get(int(unit_id))
        if rec is None:
            return None
        return self._coerce_unit_view(rec)

    def get_unit_health(self, unit_id: int) -> float:
        unit = self.get_unit_by_id(unit_id)
        return 0.0 if unit is None else float(unit.health)

    def get_unit_max_health(self, unit_id: int) -> float:
        unit = self.get_unit_by_id(unit_id)
        return 0.0 if unit is None else float(unit.max_health)

    def get_unit_type(self, unit_id: int) -> Optional[int]:
        unit = self.get_unit_by_id(unit_id)
        return None if unit is None else int(unit.unit_def_id)

    def get_unit_type_name(self, unit_id: int) -> Optional[str]:
        unit = self.get_unit_by_id(unit_id)
        return None if unit is None else unit.unit_def_name

    def get_unit_position(self, unit_id: int):
        unit = self.get_unit_by_id(unit_id)
        if unit is None:
            return None
        return unit.pos

    def get_unit_team(self, unit_id: int) -> Optional[int]:
        unit = self.get_unit_by_id(unit_id)
        return None if unit is None else int(unit.team_id)

    def get_unit_ally_team(self, unit_id: int) -> Optional[int]:
        unit = self.get_unit_by_id(unit_id)
        return None if unit is None else int(unit.ally_team_id)

    def get_unit_sight_radius(self, unit_id: int) -> float:
        unit = self.get_unit_by_id(unit_id)
        return 0.0 if unit is None else float(unit.los_radius)

    def get_units_for_team(self, team_id: int) -> List[BARUnitView]:
        return [
            u for u in self.get_all_units().values()
            if u.team_id == team_id and not u.is_dead
        ]

    def get_units_for_ally_team(self, ally_team_id: int) -> List[BARUnitView]:
        return [
            u for u in self.get_all_units().values()
            if u.ally_team_id == ally_team_id and not u.is_dead
        ]

    def get_alive_units(self) -> List[BARUnitView]:
        return [u for u in self.get_all_units().values() if not u.is_dead]

    def get_units_in_radius(self, center_xyz, radius: float) -> List[BARUnitView]:
        cx, cy, cz = center_xyz
        r2 = float(radius) * float(radius)

        out: List[BARUnitView] = []
        for u in self.get_alive_units():
            dx = u.pos_x - cx
            dy = u.pos_y - cy
            dz = u.pos_z - cz
            d2 = dx * dx + dy * dy + dz * dz
            if d2 <= r2:
                out.append(u)
        return out

    def get_enemy_units_in_sight(self, unit_id: int) -> List[BARUnitView]:
        me = self.get_unit_by_id(unit_id)
        if me is None or me.is_dead:
            return []

        candidates = self.get_units_in_radius(me.pos, me.los_radius)
        return [
            u for u in candidates
            if u.ally_team_id != me.ally_team_id and u.unit_id != me.unit_id
        ]

    def get_ally_units_in_sight(self, unit_id: int) -> List[BARUnitView]:
        me = self.get_unit_by_id(unit_id)
        if me is None or me.is_dead:
            return []

        candidates = self.get_units_in_radius(me.pos, me.los_radius)
        return [
            u for u in candidates
            if u.ally_team_id == me.ally_team_id and u.unit_id != me.unit_id
        ]