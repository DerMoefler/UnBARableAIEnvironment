from typing import Any, Dict, Optional, Tuple
from engine_session import EngineSession, EngineSessionConfig


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

        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y 
            sight_range = self.unit_sight_range(agent_id)
        
        np.append(local_obs)
        return local_obs

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs 