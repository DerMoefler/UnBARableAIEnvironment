from typing import Any, Dict, Optional, Tuple, List
from src.environment.engine_session import EngineSession, EngineSessionConfig
from src.environment.reward import RewardCalculator
from src.environment.unit_data_memory import UnitMemory
<<<<<<< HEAD
from src.environment.episode_outcome import EpisodeOutcome, TERMINATED_STATUSES, TRUNCATED_STATUSES
=======
from src.utils.logger_factory import get_file_logger

import json
>>>>>>> 2f71007 (reverted old approch to parallel envs, changed logging system into severeal files, added a pool outline)
import numpy as np
import bar_ai

from pathlib import Path
from datetime import datetime

from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class EngineSnapshot:
    units: List[bar_ai.UnitData]
    status: bar_ai.EngineStatus

class BAR_Environment:
    def __init__(
        self,
        *,
        training_team_id: int,
        session_cfg: Optional[EngineSessionConfig] = None,
        max_episode_steps: int = 512,
        reward_calculator: Optional[RewardCalculator] = None,
        config_path: Optional[Path] = None,
        environment_id: Optional[int] = 0
    ):
        self.session_cfg = session_cfg or EngineSessionConfig()
        self.session: Optional[EngineSession] = None
        if config_path is not None:
            with open(config_path,"r") as config_data:
                self.data = json.load(config_data)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.logger = get_file_logger("environment{environment_id}", Path(f"src/train/experiments/{self.data["run_name"]}/{self.data["run_name"]}_environment{environment_id}_{timestamp}.log"))

        # Episode tracking
        # max_episode_steps begrenzt, wie viele RL-Steps eine Episode maximal laufen darf.
        # Wird dieses Limit erreicht, ist die Episode truncated.
        self.max_episode_steps = max_episode_steps

        # Zähler für aktuelle Episode
        self.episode_step = 0

        self.training_team_id = training_team_id


        self.reward_calculator = (
            reward_calculator if reward_calculator is not None else RewardCalculator()
        )

        self.current_update_id = 0
        self.terminated = False
        self.truncated = False
        

        self.shared_memory_name = ""

        self.unit_memory = UnitMemory(
            self.training_team_id
        )


    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        self.terminated = False
        self.truncated = False
        if self.session == None:
            self.session = EngineSession(self.session_cfg, self.end_of_session)
            self.session.start()
            self.shared_memory_name = "/sm_proc_" + str(self.session.proc.pid)
            shared_memory = bar_ai.SharedMemory.create(self.shared_memory_name)
            self.current_update_id = 0
        
<<<<<<< HEAD
        logger.info("Engine session started. Waiting for first handleEventUpdate...")
=======
        self.logger.info("Engine session started. Waiting for first handleEventUpdate...")
>>>>>>> 2f71007 (reverted old approch to parallel envs, changed logging system into severeal files, added a pool outline)

        wait_status, update_id = self.session.grpc_server.wait_for_next_update(
            previous_count=self.current_update_id,
            timeout=600.0,
        )
        if wait_status == "timeout":
            raise TimeoutError("Kein erstes handleEventUpdate nach reset(). Timeout.")

        if wait_status == "stopped":
            raise RuntimeError("gRPC server stopped unexpectedly after reset().")

        self.current_update_id = update_id

        # Episode-Zähler zurücksetzen
        self.episode_step = 0

        snapshot = self._read_engine_snapshot()
                
        self.evaluate_engine_status(snapshot.status)

        

        self.unit_memory.reset(
            snapshot.units,
            step=self.episode_step,
        )

        stats = self._get_team_stats(snapshot.units)
        self.reward_calculator.reset(stats)

        info = {}
        # Zusätzliche Debug-Informationen zurückgeben
        info["episode_step"] = self.episode_step
        info["max_episode_steps"] = self.max_episode_steps
        info["reward_state_initialized"] = self.reward_calculator.initialized

        observation = self.create_observation_dictionary(snapshot.units)
        bar_ai.SharedMemory.remove(self.shared_memory_name)

        return observation, info

    def step(self, actions: List[bar_ai.Action]):

        shared_memory = bar_ai.SharedMemory.create(self.shared_memory_name)

        for action in actions:
            shared_memory.write_action(action)
        
        # 1) das aktuelle offene Update freigeben
        self.session.grpc_server.ack_update(self.current_update_id)

        # 2) auf das nächste Update warten
        wait_status, next_update_id = self.session.grpc_server.wait_for_next_update(
            previous_count=self.current_update_id,
            timeout=60.0,
        )
        if wait_status == "timeout":
            raise TimeoutError("Kein neues handleEventUpdate nach step().")
        if wait_status == "stopped":
            raise RuntimeError("gRPC server stopped unexpectedly after reset().")

        self.current_update_id = next_update_id


        # Ein RL-Step wurde ausgeführt
        self.episode_step += 1

        snapshot = self._read_engine_snapshot()

        outcome = self.evaluate_engine_status(snapshot.status)

        if not self.terminated and self.episode_step >= self.max_episode_steps:
            self.truncated = True

        self.unit_memory.update(
            snapshot.units,
            step=self.episode_step,
        )

        stats = self._get_team_stats(snapshot.units)


        # Natural episode end:
        # z. B. alle Gegner tot oder alle eigenen Units tot.
        terminated = self.terminated

        # Artificial episode end:
        # z. B. Step-Limit oder Frame-Limit erreicht.
        # Wenn terminated True ist, soll truncated False bleiben.
        truncated = False if terminated else self.truncated

        reward = self.reward_calculator.calculate(
                    stats=stats,
                    terminated=terminated,
                    truncated=truncated,
                    outcome=outcome,
                )

        observation = self.create_observation_dictionary(snapshot.units)
         
        if terminated or truncated:
            bar_ai.SharedMemory.remove(self.shared_memory_name)
            shared_memory = bar_ai.SharedMemory.create(self.shared_memory_name)
            self.session.grpc_server.ack_update(self.current_update_id)

        info = {
            "episode_step": self.episode_step,
            "own_alive_count": stats["own_alive_count"],
            "enemy_alive_count": len(self.unit_memory.get_alive_units()),
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
        
        if not terminated and not truncated:
            bar_ai.SharedMemory.remove(self.shared_memory_name)

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
        try:
            bar_ai.SharedMemory.remove(self.shared_memory_name)
        except Exception:
            pass
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

    def _get_team_stats(
        self,
        current_units: List[bar_ai.UnitData],
    ) -> Dict[str, Any]:

        own_alive = [
            unit
            for unit in current_units
            if int(unit.team_id) == self.training_team_id
            and not bool(unit.is_dead)
            and float(unit.health) > 0.0
        ]

        remembered_enemy_alive = self.unit_memory.get_alive_units()

        return {
            "own_health_sum": float(sum(
                max(0.0, float(unit.health))
                for unit in own_alive
            )),
            "enemy_health_sum": float(sum(
                max(0.0, float(unit.health))
                for unit in remembered_enemy_alive
            )),
            "own_alive_count": len(own_alive),
            "enemy_alive_count": len(remembered_enemy_alive),

            "enemy_visible_count": (
                self.unit_memory.visible_enemy_count
            ),
            "enemy_hidden_count": (
                self.unit_memory.hidden_enemy_count
            ),
            "enemy_remembered_count": (
                self.unit_memory.remembered_enemy_count
            ),
        }

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
        A dictionary with numpy arraays with all info is being created. The np arrays 

        Parameters
        ----------
        unit_list: List[bar_ai.UnitData]
            the units in read from the shared memory
        
        Return
        ------
        dictionary : dict
            a dict where every agent is a key and the data is a numpy array with the infos from the UniData
        """
<<<<<<< HEAD
        logger.debug(unit_list)
=======
        unit_list = shared_memory.read_all_units()
        self.logger.debug(unit_list)
>>>>>>> 2f71007 (reverted old approch to parallel envs, changed logging system into severeal files, added a pool outline)
        dictionary = {f"agent_{n}": np.asarray(agent) for n, agent in enumerate(unit_list)}
        return dictionary

    def _read_engine_snapshot(self) -> EngineSnapshot:
        shared_memory = bar_ai.SharedMemory.open(
            self.shared_memory_name
        )

        units = shared_memory.read_all_units()
        statuses = shared_memory.read_all_engine_statuses()

        if len(statuses) != 1:
            status_names = [
                getattr(status, "name", str(status))
                for status in statuses
            ]

<<<<<<< HEAD
            logger.error(
                "Expected exactly one engine status, "
                "but received %d: %s",
                len(statuses),
                status_names,
=======
        truncation_reasons = status_set & TRUNCATED_STATUSES
        termination_reasons = status_set & TERMINATED_STATUSES

        # Technische Fehler haben Vorrang.
        if truncation_reasons:
            self.truncated = True

            self.logger.error(
                "end_reason=%s engine_statuses=%s error_statuses=%s",
                "ENGINE_ERROR",
                [status.name for status in statuses],
                [status.name for status in truncation_reasons],
>>>>>>> 2f71007 (reverted old approch to parallel envs, changed logging system into severeal files, added a pool outline)
            )

            raise RuntimeError(
                "Expected exactly one engine status in shared memory, "
                f"but received {len(statuses)}: {status_names}"
            )

        return EngineSnapshot(
            units=units,
            status=statuses[0],
        )

    def evaluate_engine_status(
        self,
        status: bar_ai.EngineStatus,
    ) -> Optional[EpisodeOutcome]:
        """Evaluate one engine status and update the episode state."""


        if status in TRUNCATED_STATUSES:
            self.truncated = True

            logger.error(
                "episode_outcome=%s engine_status=%s",
                None,
                getattr(status, "name", str(status)),
            )

            return None

        if status == bar_ai.EngineStatus.GAME_ENDED:
            self.terminated = True
            outcome = EpisodeOutcome.WON

<<<<<<< HEAD
            logger.info(
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

            logger.info(
                "episode_outcome=%s engine_status=%s",
                outcome.value,
                status.name,
=======
            self.logger.info(
                "end_reason=%s engine_statuses=%s termination_statuses=%s",
                "GAME_TERMINATED",
                [status.name for status in statuses],
                [status.name for status in termination_reasons],
            )

        # Alle Engines laufen normal.
        elif statuses and all(
            status == bar_ai.EngineStatus.RUNNING
            for status in statuses
        ):
            self.logger.debug(
                "end_reason=%s engine_statuses=%s",
                "RUNNING",
                [status.name for status in statuses],
>>>>>>> 2f71007 (reverted old approch to parallel envs, changed logging system into severeal files, added a pool outline)
            )

            return outcome

<<<<<<< HEAD
        if status == bar_ai.EngineStatus.RUNNING:
            logger.debug(
                "episode_outcome=%s engine_status=%s",
                None,
                status.name,
=======
            self.logger.error(
                "end_reason=%s engine_statuses=%s",
                "UNKNOWN_ENGINE_STATUS",
                [getattr(status, "name", str(status)) for status in statuses],
>>>>>>> 2f71007 (reverted old approch to parallel envs, changed logging system into severeal files, added a pool outline)
            )

            return None

        self.truncated = True

        logger.error(
            "episode_outcome=%s engine_status=%s reason=unhandled_status",
            None,
            getattr(status, "name", str(status)),
        )

        return 