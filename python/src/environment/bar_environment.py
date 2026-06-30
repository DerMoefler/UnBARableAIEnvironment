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
        # Laut aktuellem Setup ist Team 0 die KI.
        self.training_team_id = 0

        # ------------------------------------------------------------------
        # Reward tracking
        # ------------------------------------------------------------------
        # Diese Werte speichern den vorherigen Zustand.
        # Der Reward wird aus der Differenz zwischen vorherigem und aktuellem
        # Zustand berechnet.
        self.prev_own_health_sum = 0.0
        self.prev_enemy_health_sum = 0.0
        self.prev_own_alive_count = 0
        self.prev_enemy_alive_count = 0

        # Gibt an, ob der Reward-State nach reset() erfolgreich initialisiert wurde.
        self.reward_state_initialized = False

        # Letzte Reward-Komponenten für Debugging im info-Dict.
        self.last_reward_info = {}

        # Reward-Koeffizienten.
        # Positive Rewards:
        # - Schaden an Gegnern
        # - Gegner töten
        # - Spiel gewinnen
        #
        # Negative Rewards:
        # - Eigener Schaden
        # - Eigene Units verlieren
        # - Spiel verlieren
        # - Zeitstrafe
        self.reward_damage_enemy_coef = 0.01
        self.reward_damage_own_coef = 0.01
        self.reward_enemy_kill = 1.0
        self.reward_own_death = 1.0
        self.reward_win = 5.0
        self.reward_loss = 5.0
        self.reward_time_penalty = 0.001

        # Reward-Clipping verhindert extrem große Werte.
        self.reward_clip_min = -10.0
        self.reward_clip_max = 10.0

        # grpc_server für handleEventUpdate() starten.
        self.current_update_id = 0
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

        update_id = self.grpc_server.wait_for_next_update(
            previous_count=0,
            timeout=30.0,
        )
        if update_id is None:
            raise TimeoutError("Kein erstes handleEventUpdate nach reset().")
        else:
            got_update = True

        self.current_update_id = update_id

        # Episode-Zähler zurücksetzen
        self.episode_step = 0

        # Start-Frame der Episode speichern.
        # Falls der Frame noch nicht gelesen werden kann, bleibt er -1.
        self.episode_start_frame = self._safe_get_world_frame(self.session)

        # TODO: hier observation aus shared memory auslesen
        observation = self.get_obs()

        # Reward-State initialisieren.
        # Das ist wichtig, damit compute_reward() später Deltas berechnen kann:
        # vorherige Gegner-HP - aktuelle Gegner-HP.
        self._init_reward_state()

        # Zusätzliche Debug-Informationen zurückgeben
        info["episode_step"] = self.episode_step
        info["episode_start_frame"] = self.episode_start_frame
        info["max_episode_steps"] = self.max_episode_steps
        info["max_episode_frames"] = self.max_episode_frames
        info["reward_state_initialized"] = self.reward_state_initialized
        info["received_handle_event_update"] = got_update


        return observation, info

    def step(self, action):
        # TODO: action -> Engine input senden
        # Später sollte hier ungefähr stehen:
        #
        # actions = np.asarray(action).reshape(-1)
        # for agent_idx, action_id in enumerate(actions):
        #     unit_id = ...
        #     self.send_action_to_engine(unit_id, int(action_id))

        # TODO: hier action in shared memory schreiben


        # 1) das aktuelle offene Update freigeben
        self.grpc_server.ack_update(self.current_update_id)

        # 2) auf das nächste Update warten
        next_update_id = self.grpc_server.wait_for_next_update(
            previous_count=self.current_update_id,
            timeout=30.0,
        )
        if next_update_id is None:
            raise TimeoutError("Kein neues handleEventUpdate nach step().")

        self.current_update_id = next_update_id

        # TODO: hier observation aus shared memory auslesen

        session = self._require_session()

        # Ein RL-Step wurde ausgeführt
        self.episode_step += 1

        # Neue Observation auslesen
        observation = self.get_obs()

        # Reward aus Damage, Kills, Deaths, Win/Loss und Time-Penalty berechnen.
        reward = self.compute_reward()

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
            "reward": reward,
            "reward_info": self.last_reward_info,
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

    def _get_team_stats(self):
        """
        Computes current team statistics used for reward calculation.

        The reward is based on changes between the last stored state and the
        current state:
        - enemy health decrease means damage dealt
        - own health decrease means damage taken
        - enemy alive count decrease means enemy kill
        - own alive count decrease means own death

        Returns
        -------
        stats : dict
            Dictionary containing current own/enemy health sums and alive counts.
        """
        own_alive, enemy_alive = self._get_own_and_enemy_alive_units()

        own_health_sum = sum(
            max(0.0, float(unit.health))
            for unit in own_alive
        )

        enemy_health_sum = sum(
            max(0.0, float(unit.health))
            for unit in enemy_alive
        )

        stats = {
            "own_health_sum": float(own_health_sum),
            "enemy_health_sum": float(enemy_health_sum),
            "own_alive_count": int(len(own_alive)),
            "enemy_alive_count": int(len(enemy_alive)),
        }

        return stats

    def _init_reward_state(self) -> None:
        """
        Initializes the reward baseline after reset().

        This function stores the current health sums and alive counts.
        Later compute_reward() compares the new state with these stored values.
        """
        stats = self._get_team_stats()

        self.prev_own_health_sum = stats["own_health_sum"]
        self.prev_enemy_health_sum = stats["enemy_health_sum"]
        self.prev_own_alive_count = stats["own_alive_count"]
        self.prev_enemy_alive_count = stats["enemy_alive_count"]

        # If both sides are zero, shared memory probably did not provide units yet.
        # In that case reward calculation should stay safe and return 0.0.
        self.reward_state_initialized = not (
            self.prev_own_alive_count == 0
            and self.prev_enemy_alive_count == 0
        )

        self.last_reward_info = {
            "reward_state_initialized": self.reward_state_initialized,
            "own_health_sum": self.prev_own_health_sum,
            "enemy_health_sum": self.prev_enemy_health_sum,
            "own_alive_count": self.prev_own_alive_count,
            "enemy_alive_count": self.prev_enemy_alive_count,
            "enemy_damage_done": 0.0,
            "own_damage_taken": 0.0,
            "enemy_kills": 0,
            "own_deaths": 0,
            "win_bonus": 0.0,
            "loss_penalty": 0.0,
            "time_penalty": 0.0,
            "raw_reward": 0.0,
            "clipped_reward": 0.0,
        }

    def compute_reward(self) -> float:
        """
        Computes the team reward for the controlled team.

        Reward components:
        ------------------
        + enemy damage dealt
        - own damage taken
        + enemy kills
        - own deaths
        + win bonus
        - loss penalty
        - small time penalty

        Returns
        -------
        reward : float
            Team reward for this environment step.
        """
        stats = self._get_team_stats()

        own_health_sum = stats["own_health_sum"]
        enemy_health_sum = stats["enemy_health_sum"]
        own_alive_count = stats["own_alive_count"]
        enemy_alive_count = stats["enemy_alive_count"]

        # If no unit data is available yet, return 0.0 and try to initialize.
        # This prevents fake win/loss rewards when shared memory is not ready.
        if own_alive_count == 0 and enemy_alive_count == 0:
            self.reward_state_initialized = False

            self.last_reward_info = {
                "reward_state_initialized": False,
                "own_health_sum": own_health_sum,
                "enemy_health_sum": enemy_health_sum,
                "own_alive_count": own_alive_count,
                "enemy_alive_count": enemy_alive_count,
                "enemy_damage_done": 0.0,
                "own_damage_taken": 0.0,
                "enemy_kills": 0,
                "own_deaths": 0,
                "win_bonus": 0.0,
                "loss_penalty": 0.0,
                "time_penalty": 0.0,
                "raw_reward": 0.0,
                "clipped_reward": 0.0,
            }

            return 0.0

        # If the reward state was not initialized yet, initialize it now.
        # This can happen if reset() was called before shared memory had unit data.
        if not self.reward_state_initialized:
            self.prev_own_health_sum = own_health_sum
            self.prev_enemy_health_sum = enemy_health_sum
            self.prev_own_alive_count = own_alive_count
            self.prev_enemy_alive_count = enemy_alive_count
            self.reward_state_initialized = True

            self.last_reward_info = {
                "reward_state_initialized": True,
                "own_health_sum": own_health_sum,
                "enemy_health_sum": enemy_health_sum,
                "own_alive_count": own_alive_count,
                "enemy_alive_count": enemy_alive_count,
                "enemy_damage_done": 0.0,
                "own_damage_taken": 0.0,
                "enemy_kills": 0,
                "own_deaths": 0,
                "win_bonus": 0.0,
                "loss_penalty": 0.0,
                "time_penalty": 0.0,
                "raw_reward": 0.0,
                "clipped_reward": 0.0,
            }

            return 0.0

        # Damage/kills/deaths are calculated as deltas from the previous step.
        enemy_damage_done = max(0.0, self.prev_enemy_health_sum - enemy_health_sum)
        own_damage_taken = max(0.0, self.prev_own_health_sum - own_health_sum)

        enemy_kills = max(0, self.prev_enemy_alive_count - enemy_alive_count)
        own_deaths = max(0, self.prev_own_alive_count - own_alive_count)

        reward = 0.0

        # Reward for damaging enemy units.
        reward += self.reward_damage_enemy_coef * enemy_damage_done

        # Penalty for taking damage.
        reward -= self.reward_damage_own_coef * own_damage_taken

        # Reward for killing enemy units.
        reward += self.reward_enemy_kill * enemy_kills

        # Penalty for losing own units.
        reward -= self.reward_own_death * own_deaths

        win_bonus = 0.0
        loss_penalty = 0.0

        # Win/loss reward.
        if enemy_alive_count == 0 and own_alive_count > 0:
            win_bonus = self.reward_win
            reward += win_bonus

        if own_alive_count == 0 and enemy_alive_count > 0:
            loss_penalty = self.reward_loss
            reward -= loss_penalty

        # Small time penalty to discourage doing nothing forever.
        time_penalty = self.reward_time_penalty
        reward -= time_penalty

        raw_reward = float(reward)

        # Clip reward to avoid unstable training from extreme values.
        clipped_reward = float(
            np.clip(
                raw_reward,
                self.reward_clip_min,
                self.reward_clip_max,
            )
        )

        # Store current values as previous values for the next step.
        self.prev_own_health_sum = own_health_sum
        self.prev_enemy_health_sum = enemy_health_sum
        self.prev_own_alive_count = own_alive_count
        self.prev_enemy_alive_count = enemy_alive_count

        # Store reward components for debugging/logging.
        self.last_reward_info = {
            "reward_state_initialized": self.reward_state_initialized,
            "own_health_sum": own_health_sum,
            "enemy_health_sum": enemy_health_sum,
            "own_alive_count": own_alive_count,
            "enemy_alive_count": enemy_alive_count,
            "enemy_damage_done": float(enemy_damage_done),
            "own_damage_taken": float(own_damage_taken),
            "enemy_kills": int(enemy_kills),
            "own_deaths": int(own_deaths),
            "win_bonus": float(win_bonus),
            "loss_penalty": float(loss_penalty),
            "time_penalty": float(time_penalty),
            "raw_reward": raw_reward,
            "clipped_reward": clipped_reward,
        }

        return clipped_reward

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

        n_agents = 3

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