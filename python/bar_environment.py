from typing import Any, Dict, Optional, Tuple
from engine_session import EngineSession, EngineSessionConfig
import numpy as np


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
        enemy_feat_size = self.get_enemy_feat_size() #get int number of max enemies in sight * features per enemy
        ally_feat_size = self.get_ally_feat_size() # get int number of max allies in sight * features per ally
        own_feat_size = self.get_own_feat_size() # get int number of features for own unit

        own_feats = np.zeros(own_feat_size, dtype=np.float32) # fill 1-D numpy array of size own_feat_size with zeros
        enemy_features = np.zeros(enemy_feat_size, dtype=np.float32)
        ally_features = np.zeros(ally_feat_size, dtype=np.float32)
        agent_id_feats = np.zeros(self.get_n_agents(), dtype=np.float32)

        unit = self.get_unit_by_id(agent_id)
        unit_type = self.get_unit_type(agent_id)
        # available_actions = self.get_available_actions(unit_type).flatten()
        health = self.get_health(agent_id)

        if health > 0:  # otherwise dead, returns all zeros
            pos_x = self.get_pos_x(agent_id)
            pos_y = self.get_pos_y(agent_id)
            pos_z = self.get_pos_z(agent_id)
            health_percentage = self.get_health_percentage(health, self.get_health_max(agent_id))
            sight_radius = self.unit_sight_radius(unit_type)


            own_feats[:7] = np.array([float(unit_type), float(pos_x), float(pos_y), float(pos_z), float(health), float(health_percentage), float(sight_radius)], dtype=np.float32)

            enemy_idx = 0
            for enemy_unit in self.get_enemy_units_in_sight(agent_id, sight_radius):
                enemy_unit_id = enemy_unit.get_unit_id(enemy_unit)
                enemy_unit_type = self.get_unit_type(enemy_unit_id)
                enemy_pos = self.get_relative_pos(agent_id, enemy_unit_id).flatten()
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
                ally_pos = self.get_relative_pos(agent_id, ally_unit_id).flatten()
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
        
    
    def get_health(unit_id):
        return 0.0
    
    def get_health_max(unit_id):
        return 100.0
    
    def get_health_percentage(health, health_max):
        return health / health_max
    
    def get_pos_x(unit_id):
        return 0 #x
    
    def get_pos_y(unit_id):
        return 0 #y
    
    def get_pos_z(unit_id):
        return 0 #z
    
    def unit_sight_radius(unit_type):
        type_map = {
             0: "pawn",
             1: "commander"
         }
        if unit_type == "pawn":
            return 7 # have to check values
        elif unit_type == "commander": # bad practice to use elif at end
            return 13
        
    def get_enemy_units_in_sight(unit_id, sight_radius):
        # for loop?
        return
    
    def get_ally_units_in_sight(unit_id, sight_radius):
        # for loop? computing all ally distances and checking if they are in sight
        list_of_allies = {1,2,3}
        return list_of_allies
    
    def get_realtive_pos(unit_id, other_unit_id):
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


    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs 