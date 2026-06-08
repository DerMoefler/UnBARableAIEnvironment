import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable, Mapping


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


@dataclass
class EngineSessionConfig:
    engine_exe: Union[str, Path] = "~/RecoilEngine/build-amd64-linux/install/spring-headless"
    write_dir: Union[str, Path] = "~/bar-data"
    config_file: Union[str, Path] = "config/test_fast.cfg"
    startscript: Union[str, Path] = "startscripts/3PawnVs3Pawn.txt"
    cwd: Union[str, Path] = "~/RecoilEngine"

    stdout: Optional[Union[int, str, Path]] = None
    stderr: Optional[Union[int, str, Path]] = None
    merge_stderr_to_stdout: bool = False
    text_mode: bool = True

    shared_memory_reader_factory: Optional[Callable[[], Any]] = None
    ipc_ready_timeout_s: float = 10.0
    ipc_poll_interval_s: float = 0.05


class EngineSession:
    def __init__(self, cfg: EngineSessionConfig):
        self.cfg = cfg

        self.engine_exe = Path(cfg.engine_exe).expanduser()
        self.write_dir = Path(cfg.write_dir).expanduser()
        self.config_file = Path(cfg.config_file).expanduser()
        self.startscript = Path(cfg.startscript).expanduser()
        self.cwd = Path(cfg.cwd).expanduser()

        self.proc: Optional[subprocess.Popen] = None
        self._stdout_handle = cfg.stdout
        self._stderr_handle = cfg.stderr
        self._ipc = None

    def _build_cmd(self) -> List[str]:
        return [
            str(self.engine_exe),
            "--write-dir", str(self.write_dir),
            "--config", str(self.config_file),
            str(self.startscript),
        ]

    def _resolve_stream(self, stream_spec):
        if stream_spec is None:
            return None
        if stream_spec == subprocess.PIPE:
            return subprocess.PIPE
        if isinstance(stream_spec, (str, Path)):
            p = Path(stream_spec).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            return open(p, "ab")
        return stream_spec

    def _connect_ipc(self) -> None:
        if self._ipc is not None:
            return

        if self.cfg.shared_memory_reader_factory is None:
            raise RuntimeError(
                "No shared_memory_reader_factory configured. "
                "Pass a reader factory in EngineSessionConfig."
            )

        self._ipc = self.cfg.shared_memory_reader_factory()

        deadline = time.time() + self.cfg.ipc_ready_timeout_s
        last_exc = None

        while time.time() < deadline:
            try:
                self._ipc.connect()
                return
            except Exception as exc:
                last_exc = exc
                time.sleep(self.cfg.ipc_poll_interval_s)

        raise RuntimeError(
            f"Failed to connect to BAR shared memory within "
            f"{self.cfg.ipc_ready_timeout_s:.2f}s"
        ) from last_exc

    def _disconnect_ipc(self) -> None:
        if self._ipc is None:
            return
        try:
            self._ipc.close()
        finally:
            self._ipc = None

    def _snapshot(self) -> Dict[str, Any]:
        self._connect_ipc()
        snapshot = self._ipc.read_snapshot()
        if not snapshot:
            raise RuntimeError("Shared memory snapshot is empty or unavailable.")
        if "units_by_id" not in snapshot:
            raise KeyError("Shared memory snapshot missing 'units_by_id'.")
        return snapshot

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

    def start(self) -> Dict[str, Any]:
        if self.is_running():
            return self.info()

        if not self.engine_exe.is_file():
            raise FileNotFoundError(f"spring-headless not found: {self.engine_exe}")
        if not os.access(self.engine_exe, os.X_OK):
            raise PermissionError(f"spring-headless is not executable: {self.engine_exe}")
        if not self.cwd.exists():
            raise FileNotFoundError(f"cwd does not exist: {self.cwd}")

        self.write_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._build_cmd()

        stdout = self._resolve_stream(self.cfg.stdout)
        stderr = self._resolve_stream(self.cfg.stderr)

        self._stdout_handle = stdout if hasattr(stdout, "close") else None
        self._stderr_handle = stderr if hasattr(stderr, "close") else None

        if self.cfg.merge_stderr_to_stdout:
            stderr = subprocess.STDOUT

        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.cwd),
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            text=self.cfg.text_mode if (stdout == subprocess.PIPE or stderr == subprocess.PIPE) else False,
            bufsize=1 if self.cfg.text_mode else 0,
        )

        self._connect_ipc()
        return self.info()

    def stop(self) -> None:
        if self.proc is None:
            self._disconnect_ipc()
            self._close_streams()
            return

        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

            for _ in range(50):
                if self.proc.poll() is not None:
                    break
                time.sleep(0.01)

            if self.proc.poll() is None:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

        self.proc = None
        self._disconnect_ipc()
        self._close_streams()

    def _close_streams(self):
        try:
            if self._stdout_handle:
                self._stdout_handle.close()
        finally:
            self._stdout_handle = None

        try:
            if self._stderr_handle and self._stderr_handle is not self._stdout_handle:
                self._stderr_handle.close()
        finally:
            self._stderr_handle = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def exit_code(self) -> Optional[int]:
        if self.proc is None:
            return None
        return self.proc.poll()

    def info(self) -> Dict[str, Any]:
        return {
            "pid": None if self.proc is None else self.proc.pid,
            "cmd": self._build_cmd(),
            "cwd": str(self.cwd),
            "write_dir": str(self.write_dir),
            "running": self.is_running(),
            "exit_code": self.exit_code(),
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    def get_world_frame(self) -> int:
        snap = self._snapshot()
        return int(snap.get("frame", -1))

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