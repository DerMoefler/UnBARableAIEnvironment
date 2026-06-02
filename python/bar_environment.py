from typing import Any, Dict, Optional, Tuple
from engine_session import EngineSession, EngineSessionConfig
import numpy as np


class BAR_Environment:
    def __init__(self, session_cfg: Optional[EngineSessionConfig] = None):
        self.session_cfg = session_cfg or EngineSessionConfig()
        self.session: Optional[EngineSession] = None

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        # Alte Session beenden
        if self.session is not None:
            self.session.stop()

        # Neue Session erstellen + starten
        self.session = EngineSession(self.session_cfg)
        info = self.session.start()

        # TODO: Observation aus Engine/Logs/IPC ableiten
        observation = None
        return observation, info

    def step(self, action):
        # TODO: action -> Engine input, obs/reward/terminated/truncated ermitteln
        observation = None
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
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
        #placeholder for all information
        enemy_feats_dim = self.get_obs_enemy_feats_size()
        ally_feats_dim = self.get_obs_ally_feats_size()
        own_feats_dim = self.get_obs_own_feats_size()

        enemy_features = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_features = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)
        agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)
        local_obs = np.array([enemy_features, ally_features, own_feats, agent_id_feats])

        unit = self.get_unit_by_id(agent_id)
        unit_type = self.get_unit_type(agent_id)
        available_actions = self.get_available_actions(unit_type)


        if unit.health > 0:  # otherwise dead, return all zeros
            health = self.get_health(agent_id)
            pos = unit.get_pos(agent_id)
            sight_radius = self.unit_sight_radius(unit_type)

            own_feats = np.concatenate((np.array([unit_type, pos, health], dtype=np.float32), available_actions.astype(np.float32)))

            idx = 0
            for enemy_unit in self.get_enemy_units_in_sight(agent_id, sight_radius):
                enemy_unit_id = enemy_unit.get_unit_id(enemy_unit)
                enemy_unit_type = enemy_unit.get_unit_type(enemy_unit_id)
                enemy_pos = self.get_relative_pos_x(agent_id, enemy_unit_id)
                enemy_health = enemy_unit_id.get_health(enemy_unit_id)
                enemy_features[idx:idx+3] = [
                    enemy_unit_type,
                    enemy_pos,
                    enemy_health
                ]
                idx += 3

            idx = 0
            for ally_unit in self.get_ally_units_in_sight(agent_id, sight_radius):
                ally_unit_id = ally_unit.get_unit_id(ally_unit)
                ally_unit_type = ally_unit.get_unit_type(ally_unit_id)
                ally_pos = self.get_relative_pos_x(agent_id, ally_unit_id)
                ally_health = ally_unit_id.get_health(ally_unit_id)

                ally_features[idx:idx+3] = [
                    ally_unit_type,
                    ally_pos,
                    ally_health
                ]
                idx += 3

        else:
            health = 0
            x = 0
            y = 0

            own_feats = np.array([unit_type, x, y, health])
        
        local_obs = np.append(own_feats.flatten(), [enemy_features.flatten(), ally_features.flatten()])

        return local_obs
    
    def get_health(self, unit_id):
        return 0

    def get_available_actions(unit_type):
        #[move_up, move_down, move_left, move_right, attack, stay, build]
        if unit_type == "pawn":
            avail_actions = np.array([1, 1, 1, 1, 1, 1, 0])
        elif unit_type == "commander":
            avail_actions = np.array([1, 1, 1, 1, 1, 1, 1])

        return avail_actions


    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs 