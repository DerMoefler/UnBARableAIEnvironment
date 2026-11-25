from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import os
import tempfile
from typing import Any, Mapping

class EngineMode(str, Enum):
    TRAINING = "training"
    PLAY_AGAINST_HUMAN = "play_against_human"

def _parse_unit_composition(
    value: Any,
    *,
    config_path: str,
) -> tuple[tuple[str, int], ...]:
    """
    Parse a JSON object such as:

        {
            "armpw": 4,
            "armrock": 2
        }

    The tuple representation is immutable and preserves
    the order from the JSON configuration.
    """
    if not isinstance(
        value,
        Mapping,
    ):
        raise ValueError(
            f"{config_path} must be a JSON object "
            "mapping unit names to counts."
        )

    if not value:
        raise ValueError(
            f"{config_path} must contain at least "
            "one unit type."
        )

    composition: list[
        tuple[str, int]
    ] = []

    for raw_unit_name, raw_count in (
        value.items()
    ):
        if not isinstance(
            raw_unit_name,
            str,
        ):
            raise ValueError(
                f"{config_path} contains a non-string "
                f"unit name: {raw_unit_name!r}."
            )

        unit_name = raw_unit_name.strip()

        if not unit_name:
            raise ValueError(
                f"{config_path} contains an empty "
                "unit name."
            )

        if any(
            character in unit_name
            for character in "\n\r;:{}|,"
        ):
            raise ValueError(
                "Invalid unit name in "
                f"{config_path}: {unit_name!r}."
            )

        if (
            not isinstance(raw_count, int)
            or isinstance(raw_count, bool)
        ):
            raise ValueError(
                f"{config_path}.{unit_name} must "
                "be an integer."
            )

        if raw_count <= 0:
            raise ValueError(
                f"{config_path}.{unit_name} must "
                "be greater than zero."
            )

        composition.append(
            (
                unit_name,
                raw_count,
            )
        )

    return tuple(
        composition
    )

@dataclass(frozen=True)
class StartScriptConfig:
    mode: EngineMode = EngineMode.TRAINING
    own_units: tuple[
        tuple[str, int],
        ...
    ] = (
        (
            "armpw",
            3,
        ),
    )

    enemy_units: tuple[
        tuple[str, int],
        ...
    ] = (
        (
            "armpw",
            3,
        ),
    )
    spawn_spacing: int = 80

    training_team_id: int = 0
    opponent_team_id: int = 1
    self_play: bool = False

    update_interval_frames: int = 2
    move_step_size: float = 100.0
    first_update_frame: int = 2

    human_player_name: str = "Human"

    @property
    def number_of_agents(
        self,
    ) -> int:
        return sum(
            count
            for _, count
            in self.own_units
        )


    @property
    def number_of_enemies(
        self,
    ) -> int:
        return sum(
            count
            for _, count
            in self.enemy_units
        )


    @property
    def own_unit_names(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            unit_name
            for unit_name, _
            in self.own_units
        )


    @property
    def enemy_unit_names(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            unit_name
            for unit_name, _
            in self.enemy_units
        )


    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> "StartScriptConfig":
        engine_config = config.get("engine", {})
        if not isinstance(
            engine_config,
            Mapping,
        ):
            raise ValueError(
                "'engine' must be a JSON object."
            )
        units_config = engine_config.get(
            "units"
        )

        if not isinstance(
            units_config,
            Mapping,
        ):
            raise ValueError(
                "'engine.units' must be a JSON object "
                "containing 'own' and 'enemy'."
            )

        own_units = _parse_unit_composition(
            units_config.get(
                "own"
            ),
            config_path="engine.units.own",
        )

        enemy_units = _parse_unit_composition(
            units_config.get(
                "enemy"
            ),
            config_path="engine.units.enemy",
        )
        ai_config = engine_config.get("ai_options", {})
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
            own_units=own_units,
            enemy_units=enemy_units,
            spawn_spacing=int(
                engine_config.get(
                    "spawn_spacing",
                    80,
                )
            ),
            training_team_id=int(
                engine_config.get(
                    "training_team_id",
                    0,
                )
            ),
            opponent_team_id=int(
                engine_config.get(
                    "opponent_team_id",
                    1,
                )
            ),
            self_play=config.get(
                "self_play",
                False,
            ),
            update_interval_frames=int(
                ai_config.get(
                    "update_interval_frames",
                    2,
                )
            ),
            move_step_size=float(
                ai_config.get(
                    "move_step_size",
                    100.0,
                )
            ),
            first_update_frame=int(
                ai_config.get(
                    "first_update_frame",
                    2,
                )
            ),
            human_player_name=human_config.get(
                "name",
                "Human",
            ),
        )

        startscript_config.validate()
        return startscript_config

    def validate(self) -> None:
        if not isinstance(self.self_play, bool):
            raise TypeError(
                "self_play must be a boolean value"
            )

        if not self.own_units:
            raise ValueError(
                "own_units must contain at least "
                "one unit type."
            )

        if not self.enemy_units:
            raise ValueError(
                "enemy_units must contain at least "
                "one unit type."
            )

        if self.number_of_agents < 1:
            raise ValueError(
                "The own-unit composition must spawn "
                "at least one unit."
            )

        if self.number_of_enemies < 1:
            raise ValueError(
                "The enemy-unit composition must spawn "
                "at least one unit."
            )

        for composition_name, composition in (
            (
                "own_units",
                self.own_units,
            ),
            (
                "enemy_units",
                self.enemy_units,
            ),
        ):
            seen_unit_names: set[str] = set()

            for unit_name, count in composition:
                if (
                    not isinstance(unit_name, str)
                    or not unit_name.strip()
                ):
                    raise ValueError(
                        f"{composition_name} contains "
                        "an invalid unit name."
                    )

                if any(
                    character in unit_name
                    for character in "\n\r;:{}|,"
                ):
                    raise ValueError(
                        f"Invalid unit name in "
                        f"{composition_name}: "
                        f"{unit_name!r}."
                    )

                if unit_name in seen_unit_names:
                    raise ValueError(
                        f"Duplicate unit name in "
                        f"{composition_name}: "
                        f"{unit_name!r}."
                    )

                seen_unit_names.add(
                    unit_name
                )

                if (
                    not isinstance(count, int)
                    or isinstance(count, bool)
                    or count <= 0
                ):
                    raise ValueError(
                        f"Invalid count for "
                        f"{composition_name}.{unit_name}: "
                        f"{count!r}."
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
        Team = {self.cfg.training_team_id};
        Version = stable;

        [Options]
        {{
            unbarable_update_interval_frames = {self.cfg.update_interval_frames};
            unbarable_move_step_size = {self.cfg.move_step_size};
            unbarable_first_update_frame = {self.cfg.first_update_frame};
        }}
    }}
"""

    def _training_opponent_section(self) -> str:
        """
        Builds the AI section for team 1.

        In self-play mode, team 1 is controlled by another UnBARableAI
        instance. Otherwise, SimpleAI remains the opponent.
        """
        if not self.cfg.self_play:
            return f"""\
    [ai1]
    {{
        Host = 0;
        IsFromDemo = 0;
        Name = "SimpleAI";
        ShortName = "SimpleAI";
        Team = {self.cfg.opponent_team_id};
    }}
"""

        return f"""\
    [ai1]
    {{
        Host = 0;
        IsFromDemo = 0;
        Name = "UnBARableAI";
        ShortName = "UnBARableAI";
        Team = {self.cfg.opponent_team_id};
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

        agent_spawn = self._format_unit_spawn(
            self.cfg.own_units
        )

        enemy_spawn = self._format_unit_spawn(
            self.cfg.enemy_units
        )

        agent_allowed = self._format_allowed_units(
            self.cfg.own_units
        )

        enemy_allowed = self._format_allowed_units(
            self.cfg.enemy_units
        )

        return f"""\
    [modoptions]
    {{
        minSpeed = {min_speed};
        maxSpeed = {max_speed};
        startSpeed = {start_speed};
        deathmode = killall;

        rl_team1_spawn = {agent_spawn};
        rl_team1_allowed = {agent_allowed};

        rl_team2_spawn = {enemy_spawn};
        rl_team2_allowed = {enemy_allowed};

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
            f"{self._training_opponent_section()}\n"
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

    @staticmethod
    def _format_unit_spawn(
        composition: tuple[
            tuple[str, int],
            ...
        ],
    ) -> str:
        """
        Convert the composition to the RLStart spawn format.

        Example:
            (("armpw", 4), ("armrock", 2))

        Result:
            armpw:4|armrock:2
        """
        return "|".join(
            f"{unit_name}:{count}"
            for unit_name, count
            in composition
        )


    @staticmethod
    def _format_allowed_units(
        composition: tuple[
            tuple[str, int],
            ...
        ],
    ) -> str:
        """
        Convert the composition to the RLStart allowlist format.

        Example:
            (("armpw", 4), ("armrock", 2))

        Result:
            armpw|armrock
        """
        return "|".join(
            unit_name
            for unit_name, _
            in composition
        )