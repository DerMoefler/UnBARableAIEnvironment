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
        # placeholder for all information
        enemy_feat_size = self.get_enemy_feat_size()
        ally_feat_size = self.get_ally_feat_size()
        own_feat_size = self.get_own_feat_size()

        own_feats = np.zeros(own_feat_size, dtype=np.float32)
        enemy_features = np.zeros(enemy_feat_size, dtype=np.float32)
        ally_features = np.zeros(ally_feat_size, dtype=np.float32)
        agent_id_feats = np.zeros(self.get_n_agents(), dtype=np.float32)

        unit = self.get_unit_by_id(agent_id)
        unit_type = self.get_unit_type(agent_id)
        available_actions = self.get_available_actions(unit_type).flatten()

        if unit.health > 0:  # otherwise dead, return all zeros
            health = self.get_health(agent_id)
            try:
                x, y = unit.get_pos(agent_id)
            except Exception:
                x = getattr(unit, "x", 0.0)
                y = getattr(unit, "y", 0.0)
            sight_radius = self.unit_sight_radius(unit_type)

            own_feats[:4] = np.array([float(unit_type), float(x), float(y), float(health)], dtype=np.float32)

            enemy_idx = 0
            for enemy_unit in self.get_enemy_units_in_sight(agent_id, sight_radius):
                enemy_unit_id = enemy_unit.get_unit_id(enemy_unit)
                enemy_unit_type = self.get_unit_type(enemy_unit_id)
                enemy_pos = self.get_relative_pos_x(agent_id, enemy_unit_id)
                enemy_health = enemy_unit_id.get_health(enemy_unit_id)

                enemy_features[enemy_idx, :3] = [
                    float(enemy_unit_type),
                    float(enemy_pos),
                    float(enemy_health)
                ]
                enemy_idx += 1

            ally_idx = 0
            for ally_unit in self.get_ally_units_in_sight(agent_id, sight_radius):
                ally_unit_id = ally_unit.get_unit_by_id(ally_unit)
                ally_unit_type = self.get_unit_type(ally_unit_id)
                ally_pos = self.get_relative_pos_x(agent_id, ally_unit_id)
                ally_health = ally_unit_id.get_health(ally_unit_id)

                ally_features[ally_idx, :3] = [
                    float(ally_unit_type),
                    float(ally_pos),
                    float(ally_health)
                ]
                ally_idx += 1


        local_obs = np.concatenate(
            (
                own_feats.flatten(),
                enemy_features.flatten(),
                ally_features.flatten(),
                agent_id_feats.flatten(),
            )
        )

        return local_obs
    
    def get_health(self, unit_id):
        return 0

    def get_available_actions(self, unit_type):
        #[move_up, move_down, move_left, move_right, attack, stay, build]
        type_map = {
            0: "pawn",
            1: "commander"
        }
        if unit_type == "pawn":
            avail_actions = np.array([1, 1, 1, 1, 1, 1, 0])
        elif unit_type == "commander":
            avail_actions = np.array([1, 1, 1, 1, 1, 1, 1])

        return avail_actions


    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs 