from typing import Any, Dict, Optional, Tuple
from src.environment.engine_session import EngineSession, EngineSessionConfig
from src.environment.grpc_server import UnBARableAIGRPCServer
import numpy as np

import bar_ai


class BAR_Environment:
    def __init__(
        self,
        session_cfg: Optional[EngineSessionConfig] = None,
        max_episode_steps: int = 512,
        max_episode_frames: Optional[int] = None,
    ):
        self.session_cfg = session_cfg or EngineSessionConfig()
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

        # Team-ID der lernenden Agenten.
        # Für erste Tests hart auf 0 gesetzt.
        # Muss später ggf. an euer Startscript angepasst werden.
        self.training_team_id = 0

        self.grpc_server = UnBARableAIGRPCServer()
        self.grpc_server.start()
        
    def _require_session(self) -> EngineSession:
        if self.session is None or not self.session.is_running():
            raise RuntimeError("Engine session is not running. Call reset() first.")
        return self.session

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        # Alte Session beenden
        if self.session is not None:
            self.session.stop()

        # Neue Session erstellen + starten
        self.session = EngineSession(self.session_cfg)
        info = self.session.start()

        # Episode-Zähler zurücksetzen
        self.episode_step = 0

        # Start-Frame der Episode speichern.
        # Falls der Frame noch nicht gelesen werden kann, bleibt er -1.
        self.episode_start_frame = self._safe_get_world_frame(self.session)

        # TODO: Observation aus Engine/Logs/IPC ableiten
        observation = self.get_obs()

        # Zusätzliche Debug-Informationen zurückgeben
        info["episode_step"] = self.episode_step
        info["episode_start_frame"] = self.episode_start_frame
        info["max_episode_steps"] = self.max_episode_steps
        info["max_episode_frames"] = self.max_episode_frames

        return observation, info

    def step(self, action):
        # TODO: action -> Engine input senden
        # Später sollte hier ungefähr stehen:
        #
        # actions = np.asarray(action).reshape(-1)
        # for agent_idx, action_id in enumerate(actions):
        #     unit_id = ...
        #     self.send_action_to_engine(unit_id, int(action_id))
        #
        # Danach sollte die Engine einige Frames weiterlaufen:
        # self.wait_for_engine_frames(num_frames=8)

        session = self._require_session()

        # Ein RL-Step wurde ausgeführt
        self.episode_step += 1

        # Neue Observation auslesen
        observation = self.get_obs()

        # TODO: Reward später durch compute_reward() ersetzen
        reward = 0.0

        # Natural episode end:
        # z. B. alle Gegner tot oder alle eigenen Units tot.
        terminated = self._is_terminal()

        # Artificial episode end:
        # z. B. Step-Limit oder Frame-Limit erreicht.
        # Wenn terminated True ist, soll truncated False bleiben.
        truncated = False if terminated else self._is_truncated()

        # Aktuelle Unit-Zahlen für Debugging
        own_alive, enemy_alive = self._get_own_and_enemy_alive_units()

        info = {
            "frame": self._safe_get_world_frame(session),
            "episode_step": self.episode_step,
            "episode_start_frame": self.episode_start_frame,
            "own_alive_count": len(own_alive),
            "enemy_alive_count": len(enemy_alive),
            "terminated": terminated,
            "truncated": truncated,
            "terminal_reason": self._get_terminal_reason() if terminated else "not_terminal",
            "truncation_reason": self._get_truncation_reason() if truncated else "not_truncated",
        }

        return observation, reward, terminated, truncated, info

    def close(self):
        if self.session is not None:
            self.session.stop()
            self.session = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_obs_agent(self, agent_id):
        """
        Returns the observation for a specific agent

        Parameters
        ----------
        self : BAR_Environment
            The training environment instance.
        agent_id : int
            The ID of the agent for which to retrieve the observation.
        
        Returns
        -------
        observation : np.ndarray
            A 1-D numpy array containing the observation for the specified agent.

        Examples
        --------
        >>> get_obs_agent(bar_env, 67)
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        0. 0. 0. 0. 0. 0. 0. 0.]
        """

        data = bar_ai.UnitData()  # should include date about every existing unit
        pawn = bar_ai.Pawn(data)  # need to reference the unit

        # placeholder for all information
        enemy_max, enemy_feat = self.get_enemy_feat_size()  # (max_enemies, features_per_enemy)
        ally_max, ally_feat = self.get_ally_feat_size()     # (max_allies, features_per_ally)
        own_feat_size = self.get_own_feat_size()            # number of features for own unit

        own_feats = np.zeros(own_feat_size, dtype=np.float32)
        enemy_features = np.zeros((enemy_max, enemy_feat), dtype=np.float32)
        ally_features = np.zeros((ally_max, ally_feat), dtype=np.float32)

        unit_type = pawn.getUnitType()
        # available_actions = self.get_available_actions(unit_type).flatten()
        health = pawn.getHealth()

        if health > 0:  # otherwise dead, returns all zeros
            pos_x = pawn.getXPosition()
            pos_y = pawn.getYPosition()
            pos_z = pawn.getZPosition()
            health_percentage = self.get_health_percentage(health, pawn.getMaxHealth())
            sight_radius = pawn.getSightRange()

            own_feats[:7] = np.array(
                [
                    float(unit_type),
                    float(pos_x),
                    float(pos_y),
                    float(pos_z),
                    float(health),
                    float(health_percentage),
                    float(sight_radius),
                ],
                dtype=np.float32,
            )

            enemy_idx = 0
            for enemy_unit in pawn.getUnitsInSight():  # pawn.getEnemyUnitsInSight(): should get an array of unit objects
                if enemy_unit.getTeam() != pawn.getTeam():
                    # Prevent writing outside the fixed observation array
                    if enemy_idx >= enemy_max:
                        break

                    enemy_unit_type = enemy_unit.getUnitType()
                    enemy_pos = np.asarray(self.get_relative_pos(pawn, enemy_unit)).flatten()
                    enemy_health = enemy_unit.getHealth()

                    enemy_features[enemy_idx, :5] = [
                        float(enemy_unit_type),
                        float(enemy_pos[0]),
                        float(enemy_pos[1]),
                        float(enemy_pos[2]),
                        float(enemy_health),
                    ]
                    enemy_idx += 1

            ally_idx = 0
            for ally_unit in pawn.getUnitsInSight():
                if ally_unit.getTeam() == pawn.getTeam():
                    # Prevent writing outside the fixed observation array
                    if ally_idx >= ally_max:
                        break

                    ally_unit_type = ally_unit.getUnitType()
                    ally_pos = np.asarray(self.get_relative_pos(pawn, ally_unit)).flatten()
                    ally_health = ally_unit.getHealth()

                    ally_features[ally_idx, :5] = [
                        float(ally_unit_type),
                        float(ally_pos[0]),
                        float(ally_pos[1]),
                        float(ally_pos[2]),
                        float(ally_health),
                    ]
                    ally_idx += 1

        local_obs = np.concatenate(
            (
                own_feats.flatten(),
                enemy_features.flatten(),
                ally_features.flatten(),
            )
        )

        return local_obs

    def get_enemy_feat_size(self):
        """
        Returns the size of enemy features, which is hardcoded for now

        Paramters
        ---------
        self: Bar_Environment
            The bar training environment
        
        Returns
        -------
        n_enemies: 3
            number of enemies
        n_enemy_features: 5
            number of enemy features
        
        Examples
        --------
        >>> get_enemy_feat_size(bar_env)
        3, 5
        """

        # Wichtig:
        # Die Reihenfolge muss zu get_obs_agent() passen:
        # enemy_max, enemy_feat = self.get_enemy_feat_size()
        n_enemies = 3
        n_enemy_features = 5

        return (n_enemies, n_enemy_features)
    
    def get_ally_feat_size(self):
        """
        Returns the size of ally features, which is hardcoded for now

        Paramters
        ---------
        self: Bar_Environment
            The bar training environment
        
        Returns
        -------
        n_allies: 2
            number of allies
        n_ally_features: 5
            number of ally features
        
        Examples
        --------
        >>> get_ally_feat_size(bar_env)
        2, 5
        """

        # Wichtig:
        # Die Reihenfolge muss zu get_obs_agent() passen:
        # ally_max, ally_feat = self.get_ally_feat_size()
        n_allies = 2
        n_ally_features = 5

        return (n_allies, n_ally_features)
    
    def get_own_feat_size(self):
        """
        Returns the number of own features, has to be changed when implementing new units

        Paramters
        ---------
        self: BAR_Environment
            The bar training environment

        Returns
        -------
        n_own_features: int
            The number of feature of the own agent
        
        Example
        -------
        >>> get_own_feat_size(bar_env)
        7
        """

        n_own_features = 7
        return n_own_features
    
    def get_health_percentage(self, health, health_max):
        """
        Returns the percentage of health a unit has.

        Paramters
        ---------
        self: BAR_Environment
            The bar training environment
        health: float_32
            The current health of a unit
        health_max: float_32
            The maximum health of a unit
        
        Returns
        -------
        health_percentage : float_32
            The percentage of health a unit thas

        Examples
        --------
        >>> get_health_percentage(self, 146.0, 200.0)
        0.73
        """

        # Prevent division by zero
        if health_max <= 0:
            return 0.0

        health_percentage = health / health_max

        return health_percentage
    
    def get_relative_pos(self, agent, second_unit_id):
        """
        Calculates the relative position of a unit to the agent unit

        Parameters
        ----------
        self : BAR_Environment
            The training environment instance
        
        Returns
        -------
        rel_pos : np.array
            A 1-D numpy array containing the relative position (x, y, z) of the second unit to the agent unit
        
        Examples
        --------
        >>> get_relative_pos(bar_env, 1, 0)
        [3.0, 5.2, 7.3]
        """
        relx = second_unit_id.getXPosition() - agent.getXPosition()
        rely = second_unit_id.getYPosition() - agent.getYPosition()
        relz = second_unit_id.getZPosition() - agent.getZPosition()
        rel_pos = np.array([relx, rely, relz], dtype=np.float32)
        return rel_pos

    def _safe_get_world_frame(self, session: EngineSession) -> int:
        """
        Safely returns the current engine frame.

        Parameters
        ----------
        session : EngineSession
            The currently running engine session.

        Returns
        -------
        frame : int
            Current world frame.
            Returns -1 if the frame cannot be read.
        """
        try:
            return int(session.get_world_frame())
        except Exception:
            return -1

    def _get_own_and_enemy_alive_units(self):
        """
        Returns alive own and enemy units.

        This function uses EngineSession.get_alive_units().
        It therefore requires the shared memory reader to be configured.

        Returns
        -------
        own_alive : list
            Alive units of self.training_team_id.
        enemy_alive : list
            Alive units of all other teams.
        """
        try:
            session = self._require_session()
            alive_units = session.get_alive_units()
        except Exception:
            # If shared memory is not ready yet, do not crash the environment.
            # In that case termination should not trigger.
            return [], []

        own_alive = [
            unit for unit in alive_units
            if int(unit.team_id) == self.training_team_id
            and not unit.is_dead
            and float(unit.health) > 0.0
        ]

        enemy_alive = [
            unit for unit in alive_units
            if int(unit.team_id) != self.training_team_id
            and not unit.is_dead
            and float(unit.health) > 0.0
        ]

        return own_alive, enemy_alive

    def _is_terminal(self):
        """
        Checks whether the episode ended naturally.

        Terminated means:
        - all own units are dead
        - or all enemy units are dead

        Returns
        -------
        terminated : bool
            True if the scenario is naturally finished.
        """
        own_alive, enemy_alive = self._get_own_and_enemy_alive_units()

        # If no unit data can be read yet, avoid ending the episode immediately.
        # This can happen while shared memory is not connected yet.
        if len(own_alive) == 0 and len(enemy_alive) == 0:
            return False

        if len(own_alive) == 0:
            return True

        if len(enemy_alive) == 0:
            return True

        return False

    def _get_terminal_reason(self):
        """
        Returns a readable reason for termination.

        Returns
        -------
        terminal_reason : str
            Reason why the episode terminated.
        """
        own_alive, enemy_alive = self._get_own_and_enemy_alive_units()

        if len(own_alive) == 0 and len(enemy_alive) == 0:
            return "no_unit_data_or_draw"

        if len(own_alive) == 0:
            return "loss_all_own_units_dead"

        if len(enemy_alive) == 0:
            return "win_all_enemy_units_dead"

        return "not_terminal"

    def _is_truncated(self):
        """
        Checks whether the episode was artificially stopped.

        Truncated means:
        - max_episode_steps reached
        - or max_episode_frames reached, if configured

        Returns
        -------
        truncated : bool
            True if the episode hit an artificial limit.
        """
        if self.episode_step >= self.max_episode_steps:
            return True

        # Frame based truncation is optional.
        # If max_episode_frames is None, only max_episode_steps is used.
        if self.max_episode_frames is None:
            return False

        session = self._require_session()
        current_frame = self._safe_get_world_frame(session)

        if self.episode_start_frame < 0:
            return False

        if current_frame < 0:
            return False

        elapsed_frames = current_frame - self.episode_start_frame

        if elapsed_frames >= self.max_episode_frames:
            return True

        return False

    def _get_truncation_reason(self):
        """
        Returns a readable reason for truncation.

        Returns
        -------
        truncation_reason : str
            Reason why the episode was truncated.
        """
        if self.episode_step >= self.max_episode_steps:
            return "max_episode_steps"

        if self.max_episode_frames is None:
            return "not_truncated"

        session = self._require_session()
        current_frame = self._safe_get_world_frame(session)

        if self.episode_start_frame >= 0 and current_frame >= 0:
            elapsed_frames = current_frame - self.episode_start_frame

            if elapsed_frames >= self.max_episode_frames:
                return "max_episode_frames"

        return "not_truncated"

    # def get_available_actions(self, unit_type):
    #     #[move_up, move_down, move_left, move_right, attack, stay, build]
    #     type_map = {
    #         0: "pawn",
    #         1: "commander"
    #     }
    #     if unit_type == "pawn":
    #         avail_actions = np.array([1, 1, 1, 1, 1, 1, 0])
    #     elif unit_type == "commander":
    #         avail_actions = np.array([1, 1, 1, 1, 1, 1, 1])
    #
    #     return avail_actions

    def get_n_agents(self):
        """
        Returns the number of agents, has to be changed fpr more agents

        Parameters
        ----------
        self : BAR_Environment
            The training environment instance
        
        Returns
        -------
        n_agnets : int
            number of agents
        
        Examples
        --------
        >>> get_n_agents(bar_env)
        1
        """

        n_agents = 2

        return n_agents

    def get_obs(self):
        """
        Returns the global observation for all agents

        Parameters
        ----------
        self : BAR_Environment
            The training environment instance

        Returns
        -------
        observations : List[np.ndarray]
            A list of 1-D numpy arrays, each containing the observation for a specific agent

        Examples
        --------
        >>> get_obs(bar_env)
        [array(
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        0. 0. 0. 0. 0. 0. 0. 0.],
        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        0. 0. 0. 0. 0. 0. 0. 0.]
        )]
        """
        agents_obs = [self.get_obs_agent(i) for i in range(self.get_n_agents())]
        return agents_obs