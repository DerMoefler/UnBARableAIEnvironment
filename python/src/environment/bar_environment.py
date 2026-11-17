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

        self.session_cfg = EngineSessionConfig(engine_id=0, startscript_config=(startscript_config), unbarable_ai_team_ids=unbarable_ai_team_ids) #TODO: later handled by engine pool
        # Episode tracking
        # max_episode_steps begrenzt, wie viele RL-Steps eine Episode maximal laufen darf.
        # Wird dieses Limit erreicht, ist die Episode truncated.
        self.max_episode_steps = max_episode_steps

        # Zähler für aktuelle Episode
        self.episode_step = 0

        self.training_team_id = self.unbarable_ai_team_ids[0]


        self.reward_calculator = (
            reward_calculator
            if reward_calculator is not None
            else RewardCalculator(
                training_team_id=self.training_team_id,
                config=reward_config,
            )
        )

        self.terminated = False
        self.truncated = False


    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        self.terminated = False
        self.truncated = False
        # TODO replace with pool
        if self.session == None:
            self.session = EngineSession(self.session_cfg, self.end_of_session)
            self.session.start()

        assert self.session.shared_memory is not None

        self.session.shared_memory.clear_units_and_actions()

        self.logger.info("Engine session started. Waiting for first handleEventUpdate...")

        self.session.wait_for_next_update(
            team_id=0,
            timeout=30.0,
        )

        # Episode-Zähler zurücksetzen
        self.episode_step = 0

        snapshot = self._read_engine_snapshot()
                
        self.evaluate_engine_status(snapshot.status)

        self.reward_calculator.reset(snapshot.units)

        info = {}
        # Zusätzliche Debug-Informationen zurückgeben
        info["episode_step"] = self.episode_step
        info["max_episode_steps"] = self.max_episode_steps
        info["reward_state_initialized"] = self.reward_calculator.initialized

        observation = self.create_observation_dictionary(snapshot.units)

        return observation, info

    def step(self, actions: List[bar_ai.Action]):
        assert self.session is not None

        # 0) Update actions in the shared-memory
        self.__update_actions(actions)


        self.session.ack_update(team_id=0)

        self.session.wait_for_next_update(
            team_id=0,
            timeout=60.0,
        )

        self.episode_step += 1

        snapshot = self._read_engine_snapshot()

        outcome = self.evaluate_engine_status(snapshot.status)

        if not self.terminated and self.episode_step >= self.max_episode_steps:
            self.truncated = True


        # Natural episode end:
        # z. B. alle Gegner tot oder alle eigenen Units tot.
        terminated = self.terminated

        # Artificial episode end:
        # z. B. Step-Limit oder Frame-Limit erreicht.
        # Wenn terminated True ist, soll truncated False bleiben.
        truncated = False if terminated else self.truncated

        reward = self.reward_calculator.calculate(
                    units=snapshot.units,
                    terminated=terminated,
                    truncated=truncated,
                    outcome=outcome,
                )

        observation = self.create_observation_dictionary(snapshot.units)

        if terminated or truncated:
            self.session.ack_update(team_id=0)

        info = {
            "episode_step": self.episode_step,
            "terminated": terminated,
            "truncated": truncated,
            "episode_outcome": (
                outcome.value
                if outcome is not None
                else None
            ),
            "reward": reward,
            "reward_info": self.reward_calculator.last_info,
        }

        return observation, reward, terminated, truncated, info

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

    def create_observation_dictionary(self, unit_list: List[bar_ai.UnitData]):
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

    def __update_actions(self, actions: List[bar_ai.Action]) -> None:
        """
        Update the actions in the shared-memory.

        Clears all action slots if there are more slots than actions provided.
        If there are too many actions, the trailing actions will not be written.
        This is also being logged.

        Parameters
        ----------
        actions: List[bar_ai.Action]
            the actions to write (update) in the shared memory
        """
        assert self.session is not None
        assert self.session.shared_memory is not None

        action_serializable_ids = self.session.shared_memory.get_action_ids()

        num_ids = len(action_serializable_ids)
        num_actions = len(actions)

        if num_actions > num_ids:
            self.logger.error(
                f"Received {num_actions} actions but only {num_ids} slots are available. "
                f"The last {num_actions - num_ids} actions will be discarded"
            )
        # elif num_actions < num_ids:
        #     self.logger.warning(
        #         f"Received only {num_actions} actions for {num_ids} slots. "
        #         f"Remaining {num_ids - num_actions} slots will be cleared."
        #     )

        # Write provided actions (if possible)
        for action_id, action in zip(action_serializable_ids, actions):
            self.session.shared_memory.update_action(action_id, action)

        # Clear reamining slots (if needed)
        for action_id in action_serializable_ids[num_actions:]:
            self.session.shared_memory.update_action(action_id, None)

    def _read_engine_snapshot(self) -> EngineSnapshot:
        assert self.session is not None
        assert self.session.shared_memory is not None

        shared_memory = self.session.get_shared_memory(
            team_id=team_id
        )

        units = shared_memory.read_all_units()
        status = shared_memory.get_engine_status()

        return EngineSnapshot(
            units=units,
            status=status,
        )

    def evaluate_engine_status(
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
