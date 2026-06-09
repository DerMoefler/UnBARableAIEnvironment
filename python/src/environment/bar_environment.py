from typing import Any, Dict, Optional, Tuple
from src.environment.engine_session import EngineSession, EngineSessionConfig
import numpy as np

import bar_ai


class BAR_Environment:
    def __init__(self, session_cfg: Optional[EngineSessionConfig] = None):
        self.session_cfg = session_cfg or EngineSessionConfig()
        self.session: Optional[EngineSession] = None
        
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

        # TODO: Observation aus Engine/Logs/IPC ableiten
        observation = get_obs()
        return observation, info

    def step(self, action):
        # TODO: action -> Engine input, obs/reward/terminated/truncated ermitteln
        observation = get_obs()
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

        data = bar_ai.UnitData()
        data.health = 100.0
        data.team = 1
        data.xPosition = 10.0
        data.yPosition = 20.0
        data.zPosition = 5.0
        data.hasCurrentCommand = True

        pawn = bar_ai.Pawn(data)

        # placeholder for all information
        enemy_max, enemy_feat = self.get_enemy_feat_size()  # (max_enemies, features_per_enemy)
        ally_max, ally_feat = self.get_ally_feat_size()    # (max_allies, features_per_ally)
        own_feat_size = self.get_own_feat_size()           # number of features for own unit

        own_feats = np.zeros(own_feat_size, dtype=np.float32)
        enemy_features = np.zeros((enemy_max, enemy_feat), dtype=np.float32)
        ally_features = np.zeros((ally_max, ally_feat), dtype=np.float32)
        agent_id_feats = np.zeros(self.get_n_agents(), dtype=np.float32)

        unit = self.get_unit_by_id(agent_id)
        unit_type = self.get_unit_type(agent_id)
        # available_actions = self.get_available_actions(unit_type).flatten()
        health = pawn.getHealth()  # self.get_health(agent_id)

        if health > 0:  # otherwise dead, returns all zeros
            pos_x = self.get_pos_x(agent_id)
            pos_y = self.get_pos_y(agent_id)
            pos_z = self.get_pos_z(agent_id)
            health_percentage = self.get_health_percentage(health, self.get_health_max(agent_id))
            sight_radius = self.unit_sight_radius(unit_type)


            own_feats[:7] = np.array([float(unit_type), float(pos_x), float(pos_y), float(pos_z), float(health), float(health_percentage), float(sight_radius)], dtype=np.float32)

            enemy_idx = 0
            for enemy_unit in self.get_enemy_units_in_sight(agent_id, sight_radius):
                # expected to yield unit ids (int) or objects with an id; adapt as needed
                try:
                    enemy_unit_id = int(enemy_unit)
                except Exception:
                    continue
                enemy_unit_type = self.get_unit_type(enemy_unit_id)
                enemy_pos = np.asarray(self.get_relative_pos(agent_id, enemy_unit_id)).flatten()
                enemy_health = self.get_health(enemy_unit_id)

                enemy_features[enemy_idx, :3] = [
                    float(enemy_unit_type),
                    float(enemy_pos[0]) if enemy_pos.size > 0 else 0.0,
                    float(enemy_health),
                ]
                enemy_idx += 1

            ally_idx = 0
            for ally_unit in self.get_ally_units_in_sight(agent_id, sight_radius):
                try:
                    ally_unit_id = int(ally_unit)
                except Exception:
                    continue
                ally_unit_type = self.get_unit_type(ally_unit_id)
                ally_pos = np.asarray(self.get_relative_pos(agent_id, ally_unit_id)).flatten()
                ally_health = self.get_health(ally_unit_id)

                ally_features[ally_idx, :3] = [
                    float(ally_unit_type),
                    float(ally_pos[0]) if ally_pos.size > 0 else 0.0,
                    float(ally_health),
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

    def get_enemy_feat_size(self):
        # return (max_enemies, features_per_enemy)
        return (5, 3)
    
    def get_ally_feat_size(self):
        # return (max_allies, features_per_ally)
        return (5, 2)
    
    def get_own_feat_size(self):
        return 7
    
    def get_unit_by_id(self, unit_id):
        return 0
    
    def get_unit_type(self, unit_id):
        return 0 #pawn or commander, check values
    
    def get_health(self, unit_id):
        return 10.0
    
    def get_health_max(self, unit_id):
        return 100.0
    
    def get_health_percentage(self, health, health_max):
        return health / health_max
    
    def get_pos_x(self, unit_id):
        return 0 #x
    
    def get_pos_y(self, unit_id):
        return 0 #y
    
    def get_pos_z(self, unit_id):
        return 0 #z
    
    def unit_sight_radius(self, unit_type):
        # Accept either integer type ids or string names.
        if isinstance(unit_type, int):
            if unit_type == 0:
                u = "pawn"
            elif unit_type == 1:
                u = "commander"
            else:
                u = None
        else:
            u = str(unit_type) if unit_type is not None else None

        if u == "pawn":
            return 429.0
        if u == "commander":
            return 450.0
        return 0
        
    def get_enemy_units_in_sight(self, unit_id, sight_radius):
        # default: no enemies in sight
        return []
    
    def get_ally_units_in_sight(self, unit_id, sight_radius):
        # default: no allies in sight
        return []
    
    def get_relative_pos(self, unit_id, second_unit_id):
        #posx1 - posx2
        #posy1 - posy2
        #posz1 - posz2
        #rel_pos = np.array(relx, rely, relz)
        return 0 #rel_pos

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

    #     return avail_actions

    def get_n_agents(self):
        return 1

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.get_n_agents())]
        return agents_obs 