from typing import Any, Dict, Optional, Tuple, List
from src.environment.engine_session import EngineSession, EngineSessionConfig
from src.environment.reward import RewardCalculator
from src.environment.unit_data_memory import UnitMemory
from src.environment.episode_outcome import EpisodeOutcome, TERMINATED_STATUSES, TRUNCATED_STATUSES
import numpy as np
import bar_ai

import logging

from dataclasses import dataclass

@dataclass(frozen=True)
class EngineSnapshot:
    units: List[bar_ai.UnitData]
    status: bar_ai.EngineStatus

class BAR_Environment:
    def __init__(
        self,
        session_cfg: Optional[EngineSessionConfig] = None,
        max_episode_steps: int = 512,
        reward_calculator: Optional[RewardCalculator] = None,
        port: int = 50051,
        engine_id: int = 0
    ):
        self.session_cfg = session_cfg or EngineSessionConfig(engine_id)
        self.session: Optional[EngineSession] = None

        # Episode tracking
        # max_episode_steps begrenzt, wie viele RL-Steps eine Episode maximal laufen darf.
        # Wird dieses Limit erreicht, ist die Episode truncated.
        self.max_episode_steps = max_episode_steps

        # Zähler für aktuelle Episode
        self.episode_step = 0


        self.reward_calculator = (
            reward_calculator if reward_calculator is not None else RewardCalculator()
        )

        self.current_update_id = 0
        self.terminated = False
        self.truncated = False

        self.port = port
        

        self.shared_memory_name = ""

        self.unit_memory = UnitMemory(
            training_team_id=self.reward_calculator.config.training_team_id #TODO: auf richtige config ändern
        )


    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        self.terminated = False
        self.truncated = False
        if self.session == None:
            self.session = EngineSession(self.session_cfg, self.end_of_session, self.port)
            self.session.start()
            self.shared_memory_name = "/sm_proc_" + str(self.session.proc.pid)
            shared_memory = bar_ai.SharedMemory.create(self.shared_memory_name)
            self.current_update_id = 0
        
        logging.info("Engine session started. Waiting for first handleEventUpdate...")

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
            "enemy_alive_count": UnitMemory.remembered_enemy_count,
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
        training_team_id = (
            self.reward_calculator.config.training_team_id
        )

        own_alive = [
            unit
            for unit in current_units
            if int(unit.team_id) == training_team_id
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
        logging.debug(unit_list)
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

            logging.error(
                "Expected exactly one engine status, "
                "but received %d: %s",
                len(statuses),
                status_names,
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

            logging.error(
                "episode_outcome=%s engine_status=%s",
                None,
                getattr(status, "name", str(status)),
            )

            return None

        if status == bar_ai.EngineStatus.GAME_ENDED:
            self.terminated = True
            outcome = EpisodeOutcome.WON

            logging.info(
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

            logging.info(
                "episode_outcome=%s engine_status=%s",
                outcome.value,
                status.name,
            )

            return outcome

        if status == bar_ai.EngineStatus.RUNNING:
            logging.debug(
                "episode_outcome=%s engine_status=%s",
                None,
                status.name,
            )

            return None

        self.truncated = True

        logging.error(
            "episode_outcome=%s engine_status=%s reason=unhandled_status",
            None,
            getattr(status, "name", str(status)),
        )

        return 