from typing import Any, Dict, Optional, Tuple
from src.environment.engine_session import EngineSession, EngineSessionConfig
from src.environment.reward import RewardCalculator
from typing import List
from src.environment.unit_data_memory import UnitMemory
from src.environment.episode_outcome import EpisodeOutcome, TERMINATED_STATUSES, TRUNCATED_STATUSES
import numpy as np
import bar_ai

import logging

class BAR_Environment:
    def __init__(
        self,
        session_cfg: Optional[EngineSessionConfig] = None,
        max_episode_steps: int = 512,
        max_episode_frames: Optional[int] = None,
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

        # Optionales Frame-Limit.
        # Wenn None, wird nur max_episode_steps genutzt.
        # Beispiel für 3 Minuten bei 30 FPS: 30 * 60 * 3 = 5400
        self.max_episode_frames = max_episode_frames

        # Zähler für aktuelle Episode
        self.episode_step = 0

        # Engine-Frame, bei dem die Episode gestartet wurde.
        # Wird in reset() gesetzt.
        self.episode_start_frame = -1

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

    def _require_session(self) -> EngineSession:
        """
        If there is no EngineSession object assigned to the BAR_Environemnt and running, an error ist thrown, otherwise the running session is being returned.

        Parameters
        ----------
        self : BAR_Environment
            the env itself

        Returns
        -------
        self.session : EngineSession
            the currently running engine session
        """
        if self.session is None or not self.session.is_running():
            raise RuntimeError("Engine session is not running. Call reset() first.")
        return self.session

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        if self.session == None:
            self.session = EngineSession(self.session_cfg, self.end_of_session, self.port)
            self.session.start()
            self.shared_memory_name = "/sm_proc_" + str(self.session.proc.pid)
            shared_memory = bar_ai.SharedMemory.create(self.shared_memory_name)
        
        logging.info("Engine session started. Waiting for first handleEventUpdate...")

        status, update_id = self.session.grpc_server.wait_for_next_update(
            previous_count=self.current_update_id,
            timeout=600.0,
        )
        if status == "timeout":
            raise TimeoutError("Kein erstes handleEventUpdate nach reset(). Timeout.")

        if status == "stopped":
            raise RuntimeError("gRPC server stopped unexpectedly after reset().")

        self.current_update_id = update_id

        # Episode-Zähler zurücksetzen
        self.episode_step = 0

        shared_memory = bar_ai.SharedMemory.open(self.shared_memory_name)
        self.evaluate_engine_statuses(shared_memory)

        units = shared_memory.read_all_units()

        self.unit_memory.reset(
            units,
            step=self.episode_step,
        )

        stats = self._get_team_stats(units)
        self.reward_calculator.reset(stats)

        info = {}
        # Zusätzliche Debug-Informationen zurückgeben
        info["episode_step"] = self.episode_step
        info["episode_start_frame"] = self.episode_start_frame
        info["max_episode_steps"] = self.max_episode_steps
        info["max_episode_frames"] = self.max_episode_frames
        info["reward_state_initialized"] = self.reward_calculator.initialized

        observation = self.create_observation_dictionary(shared_memory)
        bar_ai.SharedMemory.remove(self.shared_memory_name)

        return observation, info

    def step(self, actions: List[bar_ai.Action]):
        session = self._require_session()

        shared_memory = bar_ai.SharedMemory.create(self.shared_memory_name)

        for action in actions:
            shared_memory.write_action(action)
        
        # 1) das aktuelle offene Update freigeben
        self.session.grpc_server.ack_update(self.current_update_id)

        # 2) auf das nächste Update warten
        status, next_update_id = self.session.grpc_server.wait_for_next_update(
            previous_count=self.current_update_id,
            timeout=60.0,
        )
        if status == "timeout":
            raise TimeoutError("Kein neues handleEventUpdate nach step().")

        self.current_update_id = next_update_id


        # Ein RL-Step wurde ausgeführt
        self.episode_step += 1

        shared_memory = bar_ai.SharedMemory.open(self.shared_memory_name)
        outcome = self.evaluate_engine_statuses(shared_memory)

        units = shared_memory.read_all_units()

        self.unit_memory.update(
            units,
            step=self.episode_step,
        )

        stats = self._get_team_stats(units)


        own_alive = [
                    unit for unit in units
                    if int(unit.team_id) == self.reward_calculator.config.training_team_id #TODO: auf richtige config ändern
                    and not unit.is_dead
                    and float(unit.health) > 0.0
                ]

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

        observation = self.create_observation_dictionary(shared_memory)
         
        if terminated or truncated:
            bar_ai.SharedMemory.remove(self.shared_memory_name)
            shared_memory = bar_ai.SharedMemory.create(self.shared_memory_name)
            self.session.grpc_server.ack_update(self.current_update_id)

        info = {
            "episode_step": self.episode_step,
            "episode_start_frame": self.episode_start_frame,
            "own_alive_count": len(own_alive),
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

    def create_observation_dictionary(self, shared_memory: bar_ai.SharedMemory):
        """
        A dictionary with numpy arraays with all info is being created. The np arrays 

        Parameters
        ----------
        shared_memory : bar_ai.SharedMemory
            the created shared Memory instance
        
        Return
        ------
        dictionary : dict
            a dict where every agent is a key and the data is a numpy array with the infos from the UniData
        """
        unit_list = shared_memory.read_all_units()
        logging.debug(unit_list)
        dictionary = {f"agent_{n}": np.asarray(agent) for n, agent in enumerate(unit_list)}
        return dictionary

    def evaluate_engine_statuses(
        self,
        shared_memory: bar_ai.SharedMemory,
    ) -> Optional[EpisodeOutcome]:
        statuses = shared_memory.read_all_engine_statuses()
        status_set = set(statuses)

        self.terminated = False
        self.truncated = False

        truncation_reasons = status_set & TRUNCATED_STATUSES

        # Technische Fehler haben Vorrang.
        if truncation_reasons:
            self.truncated = True

            logging.error(
                "episode_outcome=%s engine_statuses=%s error_statuses=%s",
                None,
                [status.name for status in statuses],
                [status.name for status in truncation_reasons],
            )

            return None

        # GAME_ENDED wird für das Training als Sieg interpretiert.
        if bar_ai.EngineStatus.GAME_ENDED in status_set:
            self.terminated = True
            outcome = EpisodeOutcome.WON

            logging.info(
                "episode_outcome=%s engine_statuses=%s",
                outcome.value,
                [status.name for status in statuses],
            )

            return outcome

        # TEAM_DIED wird als Niederlage interpretiert.
        if ( bar_ai.EngineStatus.TEAM_DIED in status_set
            or bar_ai.EngineStatus.AI_KILLED in status_set
        ):
            self.terminated = True
            outcome = EpisodeOutcome.LOST

            logging.info(
                "episode_outcome=%s engine_statuses=%s",
                outcome.value,
                [status.name for status in statuses],
            )

            return outcome

        # Alle Engines laufen normal.
        if statuses and all(
            status == bar_ai.EngineStatus.RUNNING
            for status in statuses
        ):
            logging.debug(
                "episode_outcome=%s engine_statuses=%s",
                None,
                [status.name for status in statuses],
            )

            return None

        # Leere Liste oder nichts.
        self.truncated = True

        logging.error(
            "episode_outcome=%s engine_statuses=%s",
            None,
            [
                getattr(status, "name", str(status))
                for status in statuses
            ],
        )

        return None