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
        unit = self.get_unit_by_id(agent_id)

        unit_type = self.get_unit_type(agent_id)
        available_actions = self.get_available_actions(unit_type)

        local_obs = np.array([unit_type] + available_actions)

        if unit.health > 0:  # otherwise dead, return all zeros
            health = self.get_health(agent_id)
            x = unit.pos.x
            y = unit.pos.y
            z =
            sight_radius = self.unit_sight_radius(unit_type)

            for enemy_unit in self.get_enemy_units_in_sight(agent_id):
                enemy_unit_id = enemy_unit.get_unit_id(enemy_unit)
                enemy_unit_type = enemy_unit.get_unit_type(enemy_unit_id)
                enemy_pos = self.get_relative_pos_x(agent_id, enemy_unit_id)
                enemy_health = enemy_unit_id.get_health(enemy_unit_id)

                local_obs = np.append([enemy_unit_type, enemy_pos, enemy_health])

            for ally_unit in self.get_ally_units_in_sight(agent_id):
                ally_unit_id = ally_unit.get_unit_id(ally_unit)
                ally_unit_type = ally_unit.get_unit_type(ally_unit_id)
                ally_pos = self.get_relative_pos_x(agent_id, ally_unit_id)
                ally_health = ally_unit_id.get_health(ally_unit_id)

                local_obs = np.append([ally_unit_type, ally_pos, ally_health])

        else:
            health = 0
            x = 0
            y = 0
            sight_range = 0
        
        local_obs = np.append(local_obs, [x, y, sight_range, health])

        return local_obs
    
    def get_health():
        return 0

    def get_available_actions(unit_type):
        if unit_type == "pawn":
            avail_actions = [move_up
                             move_down,
                             move_left,
                             move_right,
                             attack,
                             stay]
        elif unit_type == "commander":
            avail_actions = ["move_up"]
        

        return avail_actions


    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs 