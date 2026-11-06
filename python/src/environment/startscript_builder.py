from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import os
import tempfile
from typing import Any, Mapping

class EngineMode(str, Enum):
    TRAINING = "training"
    PLAY_AGAINST_HUMAN = "play_against_human"


@dataclass(frozen=True)
class StartScriptConfig:
    mode: EngineMode = EngineMode.TRAINING

    number_of_agents: int = 3
    unit: str = "armpw"
    spawn_spacing: int = 80

    update_interval_frames: int = 2
    move_step_size: float = 100.0
    first_update_frame: int = 2

    human_player_name: str = "Human"

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> "StartScriptConfig":
        engine_config = config.get("engine", {})
        ai_config = engine_config.get("unbarable_ai", {})
        human_config = engine_config.get("human_player", {})

        raw_mode = engine_config.get(
            "mode",
            EngineMode.TRAINING.value,
        )

        try:
            mode = EngineMode(raw_mode)
        except ValueError as exc:
            allowed_modes = ", ".join(
                mode.value for mode in EngineMode
            )
            raise ValueError(
                f"Invalid engine.mode {raw_mode!r}. "
                f"Allowed values: {allowed_modes}"
            ) from exc

        startscript_config = cls(
            mode=mode,
            number_of_agents=config["number_of_agents"],
            unit=engine_config.get("unit", "armpw"),
            spawn_spacing=engine_config.get(
                "spawn_spacing",
                80,
            ),
            update_interval_frames=ai_config.get(
                "update_interval_frames",
                2,
            ),
            move_step_size=ai_config.get(
                "move_step_size",
                100.0,
            ),
            first_update_frame=ai_config.get(
                "first_update_frame",
                2,
            ),
            human_player_name=human_config.get(
                "name",
                "Human",
            ),
        )

        startscript_config.validate()
        return startscript_config

    def validate(self) -> None:
        if self.number_of_agents < 1:
            raise ValueError(
                "number_of_agents must be at least 1"
            )

        if not self.unit or not self.unit.strip():
            raise ValueError("unit must not be empty")

        if any(char in self.unit for char in "\n\r;:{}"):
            raise ValueError(
                f"Invalid unit name for startscript: {self.unit!r}"
            )

        if self.spawn_spacing <= 0:
            raise ValueError(
                "spawn_spacing must be greater than 0"
            )

        if self.update_interval_frames < 1:
            raise ValueError(
                "update_interval_frames must be at least 1"
            )

        if self.first_update_frame < 0:
            raise ValueError(
                "first_update_frame must not be negative"
            )

        if self.move_step_size <= 0:
            raise ValueError(
                "move_step_size must be greater than 0"
            )

        if (
            not self.human_player_name
            or any(
                char in self.human_player_name
                for char in '\n\r;{}"'
            )
        ):
            raise ValueError(
                f"Invalid human player name: "
                f"{self.human_player_name!r}"
            )


class StartScriptBuilder:
    """
    Creates a BAR/Recoil startscript for one EngineSession.

    Only selected gameplay values are configurable. Engine-critical values,
    such as the map, speed settings, start positions and team assignments,
    are fixed in the templates below.
    """

    def __init__(self, cfg: StartScriptConfig):
        cfg.validate()
        self.cfg = cfg

    def build(self) -> str:
        if self.cfg.mode == EngineMode.TRAINING:
            return self._build_training_script()

        if self.cfg.mode == EngineMode.PLAY_AGAINST_HUMAN:
            return self._build_human_script()

        raise ValueError(
            f"Unsupported engine mode: {self.cfg.mode}"
        )

    def write(self, destination: Path) -> Path:
        """
        Writes the startscript atomically.

        Atomic writing prevents the engine from observing a partially written
        startscript if sessions are started concurrently.
        """
        destination = destination.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        content = self.build()

        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            text=True,
        )

        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())

            os.replace(tmp_name, destination)
        except Exception:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise

        return destination

    def _common_sections(self) -> str:
        return f"""\
    [allyTeam0]
    {{
        startrectright = 0.2;
        startrectbottom = 1;
        startrectleft = 0;
        numallies = 0;
        startrecttop = 0;
    }}

    [allyTeam1]
    {{
        startrectright = 1;
        startrectbottom = 1;
        startrectleft = 0.8;
        numallies = 0;
        startrecttop = 0;
    }}

    [team0]
    {{
        Side = Cortex;
        Handicap = 0;
        RgbColor = 1.0 0.2 0.2;
        AllyTeam = 0;
        TeamLeader = 0;
    }}

    [team1]
    {{
        Side = Cortex;
        Handicap = 0;
        RgbColor = 0.2 0.4 1.0;
        AllyTeam = 1;
        TeamLeader = 0;
    }}

    [ai0]
    {{
        Host = 0;
        IsFromDemo = 0;
        Name = "UnBARableAI";
        ShortName = "UnBARableAI";
        Team = 0;
        Version = stable;

        [Options]
        {{
            unbarable_update_interval_frames = {self.cfg.update_interval_frames};
            unbarable_move_step_size = {self.cfg.move_step_size};
            unbarable_first_update_frame = {self.cfg.first_update_frame};
        }}
    }}
"""

    def _modoptions(self, training: bool) -> str:
        if training:
            min_speed = "10000"
            max_speed = "100000"
            start_speed = "10000"
        else:
            min_speed = "0.1"
            max_speed = "20"
            start_speed = "1"

        spawn = (
            f"{self.cfg.unit}:{self.cfg.number_of_agents}"
        )

        return f"""\
    [modoptions]
    {{
        minSpeed = {min_speed};
        maxSpeed = {max_speed};
        startSpeed = {start_speed};
        deathmode = killall;

        rl_team1_spawn = {spawn};
        rl_team1_allowed = {self.cfg.unit};

        rl_team2_spawn = {spawn};
        rl_team2_allowed = {self.cfg.unit};

        rl_spawn_spacing = {self.cfg.spawn_spacing};
        rl_remove_initial = all;
        rl_remove_decorations = 0;
        rl_enforce_allowed = 1;
    }}
"""

    def _build_training_script(self) -> str:
        return (
            "[Game]\n"
            "{\n"
            f"{self._common_sections()}\n"
            """\
    [ai1]
    {
        Host = 0;
        IsFromDemo = 0;
        Name = "SimpleAI";
        ShortName = "SimpleAI";
        Team = 1;
    }

"""
            f"{self._modoptions(training=True)}\n"
            """\
    [player0]
    {
        IsFromDemo = 0;
        Name = Trainer;
        Team = 0;
        Spectator = 1;
        rank = 0;
    }

    hostip = 127.0.0.1;
    hostport = 0;
    numplayers = 1;
    numusers = 5;
    startpostype = 2;
    mapname = map;
    gametype = BAR;
    ishost = 1;
    GameStartDelay = 5;
    nohelperais = 0;
    myplayername = Trainer;
    NoLuaUI = 1;
    NoLuaIntro = 1;
}
"""
        )

    def _build_human_script(self) -> str:
        player_name = self.cfg.human_player_name

        return (
            "[Game]\n"
            "{\n"
            f"{self._common_sections()}\n"
            f"""\
    [player0]
    {{
        IsFromDemo = 0;
        Name = "{player_name}";
        Team = 1;
        Spectator = 0;
        rank = 0;
    }}

"""
            f"{self._modoptions(training=False)}\n"
            f"""\
    hostip = 127.0.0.1;
    hostport = 0;
    numplayers = 1;
    numusers = 2;
    startpostype = 2;
    mapname = map;
    gametype = BAR;
    ishost = 1;
    GameStartDelay = 5;
    nohelperais = 0;
    myplayername = "{player_name}";
    NoLuaUI = 0;
    NoLuaIntro = 0;
}}
"""
        )