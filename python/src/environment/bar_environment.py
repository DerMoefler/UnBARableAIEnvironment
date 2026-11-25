from collections.abc import Mapping
import logging
from typing import Any, Dict, Optional, Tuple, List
from src.environment.engine_session import EngineSession, EngineSessionConfig
from src.environment.episode_outcome import EpisodeOutcome, RUNNING_STATUSES, TRUNCATED_STATUSES
from src.environment.reward import RewardCalculator, RewardConfig
from src.utils.logger_factory import get_file_logger, get_null_logger
from src.environment.startscript_builder import StartScriptConfig

import json
import numpy as np
import bar_ai

from pathlib import Path
from datetime import datetime

from dataclasses import dataclass

@dataclass(frozen=True)
class EngineSnapshot:
    units: List[bar_ai.UnitData]
    status: bar_ai.EngineStatus

class BAR_Environment:
    def __init__(
        self,
        *,
        max_episode_steps: int = 512,
        reward_config: Optional[RewardConfig] = None,
        startscript_config: Optional[StartScriptConfig] = None,
        run_name: Optional[str] = None,
        environment_id: int = 0,
        environment_log_level: int = logging.INFO,
    ):
        if max_episode_steps <= 0:
            raise ValueError(
                "max_episode_steps must be greater than 0."
            )

        self.session: Optional[EngineSession] = None

        logger_name = (
            f"environment{environment_id}"
        )

        if run_name is not None:
            timestamp = datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S"
            )

            self.logger = get_file_logger(
                logger_name,
                Path(
                    f"src/train/experiments/{run_name}/"
                    f"{run_name}_environment"
                    f"{environment_id}_{timestamp}.log"
                ),
                level=environment_log_level,
            )
        else:
            self.logger = get_null_logger(
                logger_name
            )

        if reward_config is None:
            reward_config = RewardConfig()

        if startscript_config is None:
            startscript_config = (
                StartScriptConfig() #TODO: later handled by engine pool
            )

        self.logger.info(
            "[REWARD CONFIG] Environment %s | "
            "Loaded values: %s",
            environment_id,
            reward_config,
        )

        if startscript_config.self_play: #TODO: later handled by engine pool
            self.unbarable_ai_team_ids = (
                startscript_config.training_team_id,
                startscript_config.opponent_team_id,
            )
        else:
            self.unbarable_ai_team_ids = (
                startscript_config.training_team_id,
            )

        self.session_cfg = EngineSessionConfig(engine_id=0, startscript_config=(startscript_config), unbarable_ai_team_ids=self.unbarable_ai_team_ids) #TODO: later handled by engine pool

        self.max_episode_steps = max_episode_steps

        self.episode_step = 0

        self.current_team_id = self.unbarable_ai_team_ids[0]
        self.current_team_index = 0

        match_team_ids = (              #TODO: later handled by engine pool
            startscript_config.training_team_id,
            startscript_config.opponent_team_id,
        )

        self.reward_calculators = {
            team_id: RewardCalculator(
                training_team_id=team_id,
                enemy_team_ids=tuple(
                    other_team_id
                    for other_team_id
                    in match_team_ids
                    if other_team_id != team_id
                ),
                config=reward_config,
            )
            for team_id
            in self.unbarable_ai_team_ids
        }

        self.reward_initialized_teams: set[int] = set()

        self.terminated = False
        self.truncated = False

    @classmethod
    def from_config(
        cls,
        config_path: Path,
        *,
        environment_id: int = 0,
        max_episode_steps: int | None = None,
    ) -> "BAR_Environment":
        """
        Create a BAR environment from the central JSON configuration.

        An explicitly supplied max_episode_steps overrides
        training.maximum_steps from the configuration.
        """
        config_path = Path(
            config_path
        ).expanduser().resolve()

        if not config_path.is_file():
            raise FileNotFoundError(
                "Environment configuration not found: "
                f"{config_path}"
            )

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as config_file:
            config = json.load(
                config_file
            )

        if not isinstance(
            config,
            Mapping,
        ):
            raise ValueError(
                "Environment configuration must "
                "be a JSON object."
            )

        run_name = config.get(
            "run_name"
        )

        if (
            not isinstance(run_name, str)
            or not run_name.strip()
        ):
            raise ValueError(
                "'run_name' must be a non-empty string."
            )

        if max_episode_steps is None:
            training_config = config.get(
                "training"
            )
    
            if not isinstance(
                training_config,
                Mapping,
            ):
                raise ValueError(
                    "Configuration must contain a "
                    "'training' JSON object."
                )
            
            if (
                "maximum_steps"
                not in training_config
            ):
                raise ValueError(
                    "Missing configuration key: "
                    "'training.maximum_steps'."
                )

            try:
                resolved_max_episode_steps = int(
                    training_config[
                        "maximum_steps"
                    ]
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "'training.maximum_steps' must "
                    "be an integer."
                ) from exc
        else:
            if (
                not isinstance(
                    max_episode_steps,
                    int,
                )
                or isinstance(
                    max_episode_steps,
                    bool,
                )
            ):
                raise ValueError(
                    "max_episode_steps must be an integer."
                )

            resolved_max_episode_steps = (
                max_episode_steps
            )

        if resolved_max_episode_steps <= 0:
            raise ValueError(
                "max_episode_steps must be greater than 0."
            )

        reward_data = config.get(
            "reward"
        )

        if reward_data is None:
            reward_config = RewardConfig()

        elif not isinstance(
            reward_data,
            Mapping,
        ):
            raise ValueError(
                "'reward' must be a JSON object."
            )

        else:
            try:
                reward_config = RewardConfig(
                    **dict(reward_data)
                )
            except TypeError as exc:
                raise ValueError(
                    "Invalid reward configuration: "
                    f"{exc}"
                ) from exc

        startscript_config = (
            StartScriptConfig.from_config(
                config
            )
        )

        logging_config = config.get(
            "logging",
            {},
        )

        if not isinstance(
            logging_config,
            Mapping,
        ):
            raise ValueError(
                "'logging' must be a JSON object."
            )

        default_log_level_name = str(
            logging_config.get(
                "level",
                "INFO",
            )
        ).upper()

        module_levels = logging_config.get(
            "modules",
            {},
        )

        if not isinstance(
            module_levels,
            Mapping,
        ):
            raise ValueError(
                "'logging.modules' must be a JSON object."
            )

        environment_log_level_name = str(
            module_levels.get(
                __name__,
                default_log_level_name,
            )
        ).upper()

        environment_log_level = getattr(
            logging,
            environment_log_level_name,
            None,
        )

        if not isinstance(
            environment_log_level,
            int,
        ):
            raise ValueError(
                "Invalid logging level for "
                f"{__name__!r}: "
                f"{environment_log_level_name!r}."
            )

        return cls(
            max_episode_steps=(
                resolved_max_episode_steps
            ),
            reward_config=reward_config,
            startscript_config=(
                startscript_config
            ),
            run_name=run_name,
            environment_id=environment_id,
            environment_log_level=environment_log_level
        )


    def reset(
        self,
    ) -> Tuple[Any, Dict[str, Any]]:
        self.terminated = False
        self.truncated = False
        self.episode_step = 0

        self.current_team_index = 0
        self.current_team_id = (
            self.unbarable_ai_team_ids[0]
        )

        self.reward_initialized_teams.clear()
        # TODO replace with pool
        if self.session is None:
            self.session = EngineSession(
                self.session_cfg,
                self.end_of_session,
            )
            self.session.start()

        self.logger.info(
            "Engine session started. Waiting for first "
            "handleEventUpdate from team %d...",
            self.current_team_id,
        )

        self.session.wait_for_next_update(
            team_id=self.current_team_id,
            timeout=60.0,
        )

        snapshot = self._read_engine_snapshot(
            team_id=self.current_team_id,
        )

        outcome = self._evaluate_engine_status(
            snapshot.status
        )

        reward_calculator = (
            self.reward_calculators[
                self.current_team_id
            ]
        )

        reward_calculator.reset(
            snapshot.units
        )

        self.reward_initialized_teams.add(
            self.current_team_id
        )

        observation = (
            self._create_observation_dictionary(
                snapshot.units
            )
        )

        info = {
            "current_team_id": self.current_team_id,
            "acting_team_id": None,
            "reward_team_id": None,
            "episode_step": self.episode_step,
            "max_episode_steps": (
                self.max_episode_steps
            ),
            "reward_state_initialized": (
                reward_calculator.initialized
            ),
            "episode_outcome": (
                outcome.value
                if outcome is not None
                else None
            ),
        }

        return observation, info

    def step(
        self,
        actions: List[bar_ai.Action],
    ):
        assert self.session is not None

        acting_team_id = self.current_team_id

        # Actions in das Shared Memory des aktuell
        # blockierenden Teams schreiben.
        self._update_actions(
            actions=actions,
            team_id=acting_team_id,
        )

        # Aktuellen Player freigeben.
        self.session.ack_update(
            team_id=acting_team_id
        )

        # Danach erreicht die Engine den nächsten
        # UnBARableAI-Callback.
        next_team_id = self._next_team_id()

        self.session.wait_for_next_update(
            team_id=next_team_id,
            timeout=60.0,
        )

        self.current_team_id = next_team_id

        # Eine vollständige Match-Runde ist abgeschlossen,
        # sobald wieder das erste Team an der Reihe ist.
        if (
            self.current_team_index == 0
        ):
            self.episode_step += 1

        snapshot = self._read_engine_snapshot(
            team_id=self.current_team_id,
        )

        outcome = self._evaluate_engine_status(
            snapshot.status
        )

        if (
            not self.terminated
            and self.episode_step
            >= self.max_episode_steps
        ):
            self.truncated = True

        terminated = self.terminated
        truncated = (
            False
            if terminated
            else self.truncated
        )

        reward_calculator = (
            self.reward_calculators[
                self.current_team_id
            ]
        )

        if (
            self.current_team_id
            not in self.reward_initialized_teams
        ):
            reward_calculator.reset(
                snapshot.units
            )

            self.reward_initialized_teams.add(
                self.current_team_id
            )

            # Beim ersten Update dieses Teams existiert
            # noch kein vorheriger Zustand für eine
            # Reward-Differenz.
            reward = 0.0
        else:
            reward = reward_calculator.calculate(
                units=snapshot.units,
                terminated=terminated,
                truncated=truncated,
                outcome=outcome,
            )

        observation = (
            self._create_observation_dictionary(
                snapshot.units
            )
        )

        info = {
            "acting_team_id": acting_team_id,
            "current_team_id": self.current_team_id,

            "reward_team_id": self.current_team_id,

            "episode_step": self.episode_step,
            "terminated": terminated,
            "truncated": truncated,
            "episode_outcome": (
                outcome.value
                if outcome is not None
                else None
            ),
            "reward": reward,
            "reward_info": (
                reward_calculator.last_info
            ),
        }

        if terminated or truncated:
            self.session.ack_update(
                team_id=self.current_team_id
            )

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        """
        Closes the engine session and assigns none to the session member

        Parameters
        ----------
        self : BAR_Environment

        Returns
        -------
        """
        if self.session is not None:
            self.session.stop()
            self.session = None

    def __del__(self):
        """
        Closes the engine session if possible using close()

        Parameters
        ----------
        self : BAR_Environment

        Returns
        -------
        """
        try:
            self.close()
        except Exception:
            pass

    def end_of_session(self, return_code: int):
        """
        is called when the engine session ends

        Parameters
        ----------
        return_code : int
            The exit code of the engine process.
        """
        if return_code == 0:
            self.terminated = True
        else:
            self.truncated = True

    def _create_observation_dictionary(self, unit_list: List[bar_ai.UnitData]):
        """
        A dictionary with numpy arrays with all info is being created. The np arrays 

        Parameters
        ----------
        unit_list: List[bar_ai.UnitData]
            the units in read from the shared memory
        
        Return
        ------
        dictionary : dict
            a dict where every agent is a key and the data is a numpy array with the infos from the UniData
        """
        self.logger.debug(unit_list)

        dictionary = {f"agent_{n}": np.asarray(agent) for n, agent in enumerate(unit_list)}
        return dictionary

    def _update_actions(
        self,
        actions: List[bar_ai.Action],
        team_id: int,
    ) -> None:
        """
        Update the actions in the shared memory belonging
        to one UnBARableAI team.
        """
        assert self.session is not None

        shared_memory = (
            self.session.get_shared_memory(
                team_id=team_id
            )
        )

        action_serializable_ids = shared_memory.get_action_ids()

        num_ids = len(
            action_serializable_ids
        )
        num_actions = len(actions)

        if num_actions > num_ids:
            self.logger.error(
                "Received %d actions for team %d but "
                "only %d slots are available. "
                "The last %d actions will be discarded.",
                num_actions,
                team_id,
                num_ids,
                num_actions - num_ids,
            )

        for action_id, action in zip(
            action_serializable_ids,
            actions,
        ):
            action_team_id = int(
                getattr(
                    action,
                    "team_id",
                    getattr(
                        action,
                        "teamId",
                        team_id,
                    ),
                )
            )

            if action_team_id != team_id:
                raise ValueError(
                    "Action belongs to the wrong team: "
                    f"expected={team_id}, "
                    f"received={action_team_id}."
                )

            shared_memory.update_action(
                action_id,
                action,
            )

        for action_id in (
            action_serializable_ids[
                num_actions:
            ]
        ):
            shared_memory.update_action(
                action_id,
                None,
            )

    def _read_engine_snapshot(
        self,
        team_id: int,
    ) -> EngineSnapshot:
        assert self.session is not None

        shared_memory = (
            self.session.get_shared_memory(
                team_id=team_id
            )
        )

        units =  shared_memory.read_all_units()
        status = shared_memory.get_engine_status()

        return EngineSnapshot(
            units=units,
            status=status,
        )

    def _evaluate_engine_status(
        self,
        status: bar_ai.EngineStatus,
    ) -> Optional[EpisodeOutcome]:
        """Evaluate one engine status and update the episode state."""
        if status in TRUNCATED_STATUSES:
            self.truncated = True

            self.logger.error(
                "episode_outcome=%s engine_status=%s",
                None,
                getattr(status, "name", str(status)),
            )

            return None

        if status == bar_ai.EngineStatus.GAME_WON:
            self.terminated = True
            outcome = EpisodeOutcome.WON

            self.logger.info(
                "episode_outcome=%s engine_status=%s",
                outcome.value,
                status.name,
            )

            return outcome

        if status == bar_ai.EngineStatus.GAME_LOST:
            self.terminated = True
            outcome = EpisodeOutcome.LOST

            self.logger.info(
                "episode_outcome=%s engine_status=%s",
                outcome.value,
                status.name,
            )

            return outcome

        if status in RUNNING_STATUSES:
            self.logger.debug(
                "episode_outcome=%s engine_status=%s",
                None,
                status.name,
            )

            return None

        self.truncated = True

        self.logger.error(
            "episode_outcome=%s engine_status=%s reason=unhandled_status",
            None,
            getattr(status, "name", str(status)),
        )

        return

    def _next_team_id(self) -> int:
        self.current_team_index = (
            self.current_team_index + 1
        ) % len(self.unbarable_ai_team_ids)

        return self.unbarable_ai_team_ids[
            self.current_team_index
        ]