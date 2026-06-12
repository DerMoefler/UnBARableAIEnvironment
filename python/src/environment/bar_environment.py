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
        observation = self.get_obs()
        return observation, info

    def step(self, action):
        # TODO: action -> Engine input, obs/reward/terminated/truncated ermitteln
        observation = self.get_obs()
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

        data = bar_ai.UnitData() #should include date about every existing unit
        pawn = bar_ai.Pawn(data) #need to reference the unit

        # placeholder for all information
        enemy_max, enemy_feat = self.get_enemy_feat_size()  # (max_enemies, features_per_enemy)
        ally_max, ally_feat = self.get_ally_feat_size()    # (max_allies, features_per_ally)
        own_feat_size = self.get_own_feat_size()           # number of features for own unit

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


            own_feats[:7] = np.array([float(unit_type), float(pos_x), float(pos_y), float(pos_z), float(health), float(health_percentage), float(sight_radius)], dtype=np.float32)

            enemy_idx = 0
            for enemy_unit in pawn.getUnitsInSight(): #pawn.getEnemyUnitsInSight(): should get an array of unit objects, as in pawn = bar_ai.Pawn(data)
                if enemy_unit.getTeam() != pawn.getTeam():
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
        # return (max_enemies, features_per_enemy), hardcoded for now
        return (5, 3)
    
    def get_ally_feat_size(self):
        # return (max_allies, features_per_ally), hardcodede for now
        return (5, 2)
    
    def get_own_feat_size(self):
        #might be changed in the future, limited to 7 right now, because of unit_type, posx, posy, posz, health, health_percentage, sight_radius
        return 7
    
    def get_health_percentage(self, health, health_max):
        return health / health_max
    
    def get_relative_pos(self, agent, second_unit_id):
        relx = second_unit_id.getXPosition() - agent.getXPosition()
        rely = second_unit_id.getYPosition() - agent.getYPosition()
        relz = second_unit_id.getZPosition() - agent.getZPosition()
        rel_pos = np.array([relx, rely, relz], dtype=np.float32)
        return rel_pos

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