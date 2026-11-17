from types import NoneType
from typing import Any, Dict, Optional, Tuple, List
from src.environment.engine_session import EngineSession, EngineSessionConfig
from src.environment.episode_outcome import EpisodeOutcome, TERMINATED_STATUSES, TRUNCATED_STATUSES
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
        reward_calculator: Optional[RewardCalculator] = None,
        config_path: Optional[Path] = None,
        environment_id: Optional[int] = 0
    ):
        self.session: Optional[EngineSession] = None  #TODO: later handled by engine pool

        logger_name = f"environment{environment_id}"
        reward_config = RewardConfig()

        if config_path is not None:
            with open(config_path, "r") as config_data:
                self.data = json.load(config_data)

            run_name = self.data["run_name"]
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            self.logger = get_file_logger(
                logger_name,
                Path(
                    f"src/train/experiments/{run_name}/"
                    f"{run_name}_environment{environment_id}_{timestamp}.log"
                )
            )

            reward_data = self.data.get("reward")

            if reward_data is None:
                self.logger.warning(
                    "No reward configuration found. Using default values: %s",
                    reward_config,
                )
            elif not isinstance(reward_data, dict):
                self.logger.warning(
                    "Invalid reward configuration. Expected a dictionary, got %s. "
                    "Using default values: %s",
                    type(reward_data).__name__,
                    reward_config,
                )
            else:
                try:
                    reward_config = RewardConfig(**reward_data)
                except TypeError as exc:
                    raise ValueError(
                        f"Invalid reward configuration in startconfig.json: {exc}"
                    ) from exc
            
            self.logger.info(
                "[REWARD CONFIG] Environment %s | Loaded values: %s",
                environment_id,
                reward_config,
            )

            startscript_config = StartScriptConfig.from_config(self.data)  #TODO: later handled by engine pool

        else:
            self.logger = get_null_logger(logger_name)
            startscript_config = StartScriptConfig()  #TODO: later handled by engine pool

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
        # Episode tracking
        # max_episode_steps begrenzt, wie viele RL-Steps eine Episode maximal laufen darf.
        # Wird dieses Limit erreicht, ist die Episode truncated.
        self.max_episode_steps = max_episode_steps

        # Zähler für aktuelle Episode
        self.episode_step = 0

        self.current_team_id = self.unbarable_ai_team_ids[0]
        self.current_team_index = 0

        self.reward_calculators: Dict[
            int,
            RewardCalculator,
        ] = {
            team_id: RewardCalculator(
                training_team_id=team_id,
                config=reward_config,
            )
            for team_id in self.unbarable_ai_team_ids
        }

        self.reward_initialized_teams: set[int] = set()

        self.terminated = False
        self.truncated = False


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
            timeout=30.0,
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

        if status == bar_ai.EngineStatus.GAME_ENDED:
            self.terminated = True
            outcome = EpisodeOutcome.WON

            self.logger.info(
                "episode_outcome=%s engine_status=%s",
                outcome.value,
                status.name,
            )

            return outcome

        if status in {
            bar_ai.EngineStatus.TEAM_DIED,
            bar_ai.EngineStatus.AI_KILLED,
        }:
            self.terminated = True
            outcome = EpisodeOutcome.LOST

            self.logger.info(
                "episode_outcome=%s engine_status=%s",
                outcome.value,
                status.name,
            )

            return outcome

        if status == bar_ai.EngineStatus.RUNNING:
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