# On-Policy SMAC II - Local Observations Code Reference

## Quick Reference: Key Code Locations

**Repository:** https://github.com/marlbenchmark/on-policy

**File:** `onpolicy/envs/starcraft2/StarCraft2v2/starcraft2.py`

---

## 1. Main Observation Retrieval in Environment Wrapper

**File:** `onpolicy/envs/starcraft2/SMACv2_modified.py`

```python
def step(self, actions):
    """A single environment step. Returns reward, terminated, info."""
    # ... execute actions ...
    
    local_obs = self.get_obs()  # ← Get local observations for all agents
    global_state = np.array([self.env.get_state_agent(agent_id) 
                            for agent_id in range(self.env.n_agents)])
    
    rewards = [[reward]] * self.env.n_agents
    dones = [...]
    infos = [...]
    avail_actions = [self.get_avail_agent_actions(i) for i in range(self.env.n_agents)]
    
    return local_obs, global_state, rewards, dones, infos, avail_actions
```

---

## 2. Core Observation Construction Function

**File:** `onpolicy/envs/starcraft2/StarCraft2v2/starcraft2.py` - Lines ~1150-1380

```python
def get_obs_agent(self, agent_id, fully_observable=False):
    """Returns observation for agent_id. The observation is composed of:
    - agent movement features (where it can move to, height information and pathing grid)
    - enemy features (available_to_attack, health, relative_x, relative_y, shield, unit_type)
    - ally features (visible, distance, relative_x, relative_y, shield, unit_type)
    - agent unit features (health, shield, unit_type)
    All of this information is flattened and concatenated into a list, in the aforementioned order.
    """
    
    unit = self.get_unit_by_id(agent_id)
    move_feats_dim = self.get_obs_move_feats_size()
    enemy_feats_dim = self.get_obs_enemy_feats_size()
    ally_feats_dim = self.get_obs_ally_feats_size()
    own_feats_dim = self.get_obs_own_feats_size()
    
    move_feats = np.zeros(move_feats_dim, dtype=np.float32)
    enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
    ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
    own_feats = np.zeros(own_feats_dim, dtype=np.float32)
    
    if unit.health > 0 and self.obs_starcraft:
        x = unit.pos.x
        y = unit.pos.y
        sight_range = self.unit_sight_range(agent_id)
        
        # MOVEMENT FEATURES
        avail_actions = self.get_avail_agent_actions(agent_id)
        for m in range(self.n_actions_move):
            move_feats[m] = avail_actions[m + 2]
        
        ind = self.n_actions_move
        if self.obs_pathing_grid:
            move_feats[ind : ind + self.n_obs_pathing] = self.get_surrounding_pathing(unit)
            ind += self.n_obs_pathing
        if self.obs_terrain_height:
            move_feats[ind:] = self.get_surrounding_height(unit)
        
        # ENEMY FEATURES
        for e_id, e_unit in self.enemies.items():
            e_x = e_unit.pos.x
            e_y = e_unit.pos.y
            dist = self.distance(x, y, e_x, e_y)
            enemy_visible = (
                self.is_position_in_cone(agent_id, e_unit.pos) if self.conic_fov
                else dist < sight_range
            )
            
            if (enemy_visible and e_unit.health > 0) or (e_unit.health > 0 and fully_observable):
                enemy_feats[e_id, 0] = avail_actions[self.n_actions_no_attack + e_id]  # shootable
                enemy_feats[e_id, 1] = dist / sight_range                              # distance
                enemy_feats[e_id, 2] = (e_x - x) / sight_range                         # relative X
                enemy_feats[e_id, 3] = (e_y - y) / sight_range                         # relative Y
                
                show_enemy = (
                    self.mask_enemies and not self.enemy_mask[agent_id][e_id]
                ) or not self.mask_enemies
                ind = 4
                
                if self.obs_all_health and show_enemy:
                    enemy_feats[e_id, ind] = e_unit.health / e_unit.health_max  # health
                    ind += 1
                if self.shield_bits_enemy > 0:
                    max_shield = self.unit_max_shield(e_unit)
                    enemy_feats[e_id, ind] = e_unit.shield / max_shield  # shield
                    ind += 1
                if self.unit_type_bits > 0 and show_enemy:
                    type_id = self.get_unit_type_id(e_unit, False)
                    enemy_feats[e_id, ind + type_id] = 1  # unit type (one-hot)
        
        # ALLY FEATURES
        al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]
        for i, al_id in enumerate(al_ids):
            al_unit = self.get_unit_by_id(al_id)
            al_x = al_unit.pos.x
            al_y = al_unit.pos.y
            dist = self.distance(x, y, al_x, al_y)
            ally_visible = (
                self.is_position_in_cone(agent_id, al_unit.pos) if self.conic_fov
                else dist < sight_range
            )
            
            if (ally_visible and al_unit.health > 0) or (al_unit.health > 0 and fully_observable):
                ally_feats[i, 0] = 1                                    # visible
                ally_feats[i, 1] = dist / sight_range                   # distance
                ally_feats[i, 2] = (al_x - x) / sight_range             # relative X
                ally_feats[i, 3] = (al_y - y) / sight_range             # relative Y
                
                ind = 4
                
                if self.obs_all_health:
                    if not self.stochastic_health:
                        ally_feats[i, ind] = al_unit.health / al_unit.health_max  # health
                        ind += 1
                    elif self.observe_teammate_health:
                        ally_feats[i, ind] = self._compute_health(agent_id=al_id, unit=al_unit)
                        ind += 1
                    elif self.zero_pad_health:
                        ind += 1
                
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(al_unit)
                    ally_feats[i, ind] = al_unit.shield / max_shield  # shield
                    ind += 1
                
                if self.stochastic_attack and self.observe_attack_probs:
                    ally_feats[i, ind] = self.agent_attack_probabilities[al_id]
                    ind += 1
                elif self.stochastic_attack and self.zero_pad_stochastic_attack:
                    ind += 1
                
                if self.stochastic_health and self.observe_teammate_health:
                    ally_feats[i, ind] = self.agent_health_levels[al_id]
                    ind += 1
                elif self.stochastic_health and self.zero_pad_health:
                    ind += 1
                
                if self.unit_type_bits > 0 and (not self.replace_teammates or self.observe_teammate_types):
                    type_id = self.get_unit_type_id(al_unit, True)
                    ally_feats[i, ind + type_id] = 1
                    ind += self.unit_type_bits
                elif self.unit_type_bits > 0 and self.zero_pad_unit_types:
                    ind += self.unit_type_bits
                
                if self.obs_last_action:
                    ally_feats[i, ind:] = self.last_action[al_id]
        
        # OWN FEATURES
        ind = 0
        if self.obs_own_health:
            if not self.stochastic_health:
                own_feats[ind] = unit.health / unit.health_max
            else:
                own_feats[ind] = self._compute_health(agent_id, unit)
            ind += 1
        
        if self.shield_bits_ally > 0:
            max_shield = self.unit_max_shield(unit)
            own_feats[ind] = unit.shield / max_shield
            ind += 1
        
        if self.stochastic_attack:
            own_feats[ind] = self.agent_attack_probabilities[agent_id]
            ind += 1
        
        if self.stochastic_health:
            own_feats[ind] = self.agent_health_levels[agent_id]
            ind += 1
        
        if self.obs_own_pos:
            own_feats[ind] = x / self.map_x
            own_feats[ind + 1] = y / self.map_y
            ind += 2
        
        if self.conic_fov:
            own_feats[ind : ind + 2] = self.fov_directions[agent_id]
            ind += 2
        
        if self.unit_type_bits > 0:
            type_id = self.get_unit_type_id(unit, True)
            own_feats[ind + type_id] = 1
    
    # CONCATENATE ALL COMPONENTS
    if self.obs_starcraft:
        agent_obs = np.concatenate((
            move_feats.flatten(),
            enemy_feats.flatten(),
            ally_feats.flatten(),
            own_feats.flatten(),
        ))
    
    # OPTIONAL: ADD TIMESTEP
    if self.obs_timestep_number:
        if self.obs_starcraft:
            agent_obs = np.append(agent_obs, self._episode_steps / self.episode_limit)
        else:
            agent_obs = np.zeros(1, dtype=np.float32)
            agent_obs[:] = self._episode_steps / self.episode_limit
    
    return agent_obs
```

---

## 3. Feature Size Calculation Functions

```python
def get_obs_move_feats_size(self):
    """Returns the size of the vector containing movement features."""
    move_feats = self.n_actions_move  # 4 (north, south, east, west)
    if self.obs_pathing_grid:
        move_feats += self.n_obs_pathing  # +8
    if self.obs_terrain_height:
        move_feats += self.n_obs_height   # +9
    return move_feats

def get_obs_enemy_feats_size(self):
    """Returns the dimensions of the matrix containing enemy features.
    Size is (n_enemies, n_features)."""
    nf_en = 4  # shootable, distance, rel_x, rel_y
    nf_en += self.unit_type_bits
    if self.obs_all_health:
        nf_en += 1 + self.shield_bits_enemy
    return self.n_enemies, nf_en

def get_obs_ally_feats_size(self):
    """Returns the dimensions of the matrix containing ally features.
    Size is ((n_agents - 1), n_features)."""
    nf_al = 4  # visible, distance, rel_x, rel_y
    nf_cap = self.get_obs_ally_capability_size()
    if self.obs_all_health:
        nf_al += 1 + self.shield_bits_ally
    if self.obs_last_action:
        nf_al += self.n_actions
    return self.n_agents - 1, nf_al + nf_cap

def get_obs_own_feats_size(self):
    """Returns the size of the vector containing the agents' own features."""
    own_feats = self.get_cap_size()
    if self.obs_own_health and self.obs_starcraft:
        own_feats += 1 + self.shield_bits_ally
    if self.conic_fov and self.obs_starcraft:
        own_feats += 2
    if self.obs_own_pos and self.obs_starcraft:
        own_feats += 2
    return own_feats

def get_obs_size(self):
    """Returns the size of the observation as a list."""
    own_feats = self.get_obs_own_feats_size()
    move_feats = self.get_obs_move_feats_size()
    n_enemies, n_enemy_feats = self.get_obs_enemy_feats_size()
    n_allies, n_ally_feats = self.get_obs_ally_feats_size()
    
    enemy_feats = n_enemies * n_enemy_feats
    ally_feats = n_allies * n_ally_feats
    all_feats = move_feats + enemy_feats + ally_feats + own_feats
    
    timestep_feats = 1 if self.obs_timestep_number else 0
    all_feats += timestep_feats
    
    return [all_feats, [n_allies, n_ally_feats], [n_enemies, n_enemy_feats], 
            [1, move_feats], [1, own_feats+timestep_feats]]
```

---

## 4. Visibility and Distance Functions

```python
@staticmethod
def distance(x1, y1, x2, y2):
    """Distance between two points."""
    return math.hypot(x2 - x1, y2 - y1)

def unit_sight_range(self, agent_id):
    """Returns the sight range for an agent."""
    sight_range_map = {
        self.stalker_id: 10,
        self.zealot_id: 9,
        self.colossus_id: 10,
        self.zergling_id: 8,
        self.baneling_id: 8,
        self.hydralisk_id: 9,
        self.marine_id: 9,
        self.marauder_id: 10,
        self.medivac_id: 11,
    }
    unit = self.agents[agent_id]
    return sight_range_map[unit.unit_type]

def is_position_in_cone(self, agent_id, pos, range="sight_range"):
    """Check if position is within agent's cone (for conic FOV)."""
    ally_pos = self.get_unit_by_id(agent_id).pos
    distance = self.distance(ally_pos.x, ally_pos.y, pos.x, pos.y)
    
    if range == "sight_range":
        unit_range = self.unit_sight_range(agent_id)
    elif range == "shoot_range":
        unit_range = self.unit_shoot_range(agent_id)
    
    if distance > unit_range:
        return False
    
    x_diff = pos.x - ally_pos.x
    x_diff = max(x_diff, EPS) if x_diff > 0 else min(x_diff, -EPS)
    obj_angle = np.arctan((pos.y - ally_pos.y) / x_diff)
    
    x = self.fov_directions[agent_id][0]
    x = max(x, EPS) if x_diff > 0 else min(x, -EPS)
    fov_angle = np.arctan(self.fov_directions[agent_id][1] / x)
    
    return np.abs(obj_angle - fov_angle) < self.conic_fov_angle / 2
```

---

## 5. Initialization in Environment Wrapper

**File:** `onpolicy/envs/starcraft2/SMACv2_modified.py`

```python
class SMACv2(StarCraftCapabilityEnvWrapper):
    def __init__(self, **kwargs):
        super(SMACv2, self).__init__(obs_last_action=False, **kwargs)
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        self.n_agents = self.env.n_agents
        
        for i in range(self.env.n_agents):
            self.action_space.append(Discrete(self.env.n_actions))
            self.observation_space.append(self.env.get_obs_size())        # ← Obs size
            self.share_observation_space.append(self.env.get_state_size())
```

---

## 6. Example: 8m Scenario Breakdown

For the "8m" map (8 Marines vs 8 Marines):

```python
env = SMACv2(map_name="8m", obs_last_action=False, obs_all_health=True)

# Environment properties:
env.n_agents = 8           # Number of allied units
env.n_enemies = 8          # Number of enemy units
env.n_actions_move = 4     # north, south, east, west
env.n_actions_no_attack = 6  # stop, + 4 move + 1 no-op = 6
env.n_actions = 14         # 6 (no attack) + 8 (attack each enemy)

# Observation breakdown:
move_features = 4          # Just movement actions
enemy_features = 8 * 5     # 8 enemies × 5 features (shootable, distance, rel_x, rel_y, health)
ally_features = 7 * 4      # 7 allies × 4 features (visible, distance, rel_x, rel_y)
own_features = 1           # Just health
timestep = 0               # Not included

total_obs_size = 4 + 40 + 28 + 1 = 73

# Getting observations:
obs, state, avail_actions = env.reset()

for episode in range(num_episodes):
    obs, state, avail_actions = env.reset()
    
    for step in range(episode_length):
        actions = [agent.get_action(obs[i]) for i in range(env.n_agents)]
        
        obs, state, rewards, dones, infos, avail_actions = env.step(actions)
        # obs: list of 8 arrays, each of size 73
        # Each obs[i] contains local observation for agent i
```

---

## 7. Special Cases and Configurations

### Stochastic Attack (Variable Attack Probability)
```python
if self.stochastic_attack:
    # Each agent has different attack success probability
    self.agent_attack_probabilities = np.zeros(self.n_agents)
    # Added to obs if observe_attack_probs=True
    ally_feats[i, ind] = self.agent_attack_probabilities[al_id]
```

### Stochastic Health (Variable Death Levels)
```python
if self.stochastic_health:
    # Each agent has different health threshold at which it dies
    self.agent_health_levels = np.zeros(self.n_agents)
    # Rescaled health: (health/health_max - health_level) / (1 - health_level)
    own_feats[ind] = self._compute_health(agent_id, unit)
```

### Unit Type Masking
```python
if self.unit_type_bits > 0:
    # For each unit, encode type as one-hot vector
    if self.map_type == "MMM":  # Marines, Marauders, Medivacs
        # type_id ∈ {0, 1, 2}
        unit_type_encoding = [0, 0, 1]  # Example: Marauder
```

### Field of View (Conic FOV)
```python
if self.conic_fov:
    # Units can only see within a cone around them
    # Direction stored as [cos(angle), sin(angle)]
    own_feats[ind : ind + 2] = self.fov_directions[agent_id]
```

---

## 8. Data Flow Summary

```
Environment Step
    ↓
env.step(actions)
    ↓
self.get_obs()  [for all agents]
    ↓
get_obs_agent(agent_id)
    ├─ Get agent's unit reference
    ├─ Compute movement features (4-21 elements)
    ├─ For each enemy:
    │   └─ Compute [shootable, distance, rel_x, rel_y, health, shield, unit_type]
    ├─ For each ally:
    │   └─ Compute [visible, distance, rel_x, rel_y, health, shield, capabilities, last_action]
    ├─ For self:
    │   └─ Compute [health, shield, attack_prob, health_level, pos_x, pos_y, fov, unit_type]
    ├─ Concatenate all components
    ├─ Add timestep (optional)
    └─ Return as np.float32 array
    ↓
[obs_agent_0, obs_agent_1, ..., obs_agent_n]  [List of numpy arrays]
```

---

## Key Files in Repository

1. **Main Environment:** `onpolicy/envs/starcraft2/StarCraft2v2/starcraft2.py`
   - Contains `get_obs_agent()` and all feature computation

2. **Environment Wrapper:** `onpolicy/envs/starcraft2/SMACv2_modified.py`
   - Wraps the base environment and provides public interface

3. **Base Wrapper:** `onpolicy/envs/starcraft2/StarCraft2v2/wrapper.py`
   - Provides capability configuration wrapper

4. **Multi-Agent Interface:** `onpolicy/envs/starcraft2/multiagentenv.py`
   - Base class for all environments

5. **SMAC V2 Library:** (external dependency)
   - Located in `smacv2` package
   - Provides actual StarCraft II game interface

---

## Constants and Ranges

```python
# Actions
self.n_actions_move = 4       # Directions
self.n_actions_no_attack = 6  # Stop + 4 moves + 1 noop (varies with FOV)
self.n_actions = 14           # n_actions_no_attack + n_enemies

# Pathing and Terrain
self.n_obs_pathing = 8        # 8 surrounding points
self.n_obs_height = 9         # 8 surrounding + center

# Unit shields
self.shield_bits_ally = 1     # For Protoss (1) or 0 for others
self.shield_bits_enemy = 1    # For Protoss (1) or 0 for others

# Unit types
self.unit_type_bits = varies  # 0 (single type) to 3 (multiple types)

# FOV (if conic_fov=True)
self.n_fov_actions = 12       # Number of FOV direction actions
self.conic_fov_angle = 2π/12 ≈ 0.524 radians

# Episode
self.episode_limit = 400      # Max steps per episode
self._episode_steps = 0       # Current step count
```

---

## Normalization Constants

```python
EPS = 1e-7  # Small epsilon to avoid division by zero

# Normalization by sight range (typically 8-11)
distance / sight_range                 → [0-1]
(x_enemy - x_agent) / sight_range      → [-1 to 1] or beyond
(y_enemy - y_agent) / sight_range      → [-1 to 1] or beyond

# Normalization by health/shield max
unit.health / unit.health_max          → [0-1]
unit.shield / max_shield               → [0-1]

# Normalization by map dimensions
x / map_x                              → [0-1]
y / map_y                              → [0-1]

# Normalization by episode length
episode_steps / episode_limit          → [0-1]
```

