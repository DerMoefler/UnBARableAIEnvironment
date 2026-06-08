# On-Policy SMAC II Local Observations Structure Analysis

Based on exploration of the `marlbenchmark/on-policy` repository, this document details how local observations are constructed in the StarCraft II multi-agent environment.

## Overview

The local observations (local_obs) in SMAC II are structured to provide each agent with their local view of the environment, including:
- Movement capabilities and surroundings (pathing grid, terrain height)
- Visible enemy units with their relative positions and attributes
- Visible ally units with their relative positions and attributes  
- Own agent's attributes and capabilities

---

## 1. Observation Construction Flow

### Entry Point: `get_obs_agent(agent_id)`
**File:** `onpolicy/envs/starcraft2/StarCraft2v2/starcraft2.py`

This is the main function that constructs local observations for a single agent:

```python
def get_obs_agent(self, agent_id, fully_observable=False):
    """Returns observation for agent_id. The observation is composed of:
    - agent movement features (where it can move to, height information and pathing grid)
    - enemy features (available_to_attack, health, relative_x, relative_y, shield, unit_type)
    - ally features (visible, distance, relative_x, relative_y, shield, unit_type)
    - agent unit features (health, shield, unit_type)
    
    All of this information is flattened and concatenated into a list, in the aforementioned order.
    """
```

---

## 2. Observation Components (Order of Concatenation)

The observation vector is built by concatenating these components in order:

### 2.1 **Movement Features** (move_feats)
```python
move_feats_dim = self.get_obs_move_feats_size()
move_feats = np.zeros(move_feats_dim, dtype=np.float32)

# Base movement actions (4 dimensions)
for m in range(self.n_actions_move):  # n_actions_move = 4
    move_feats[m] = avail_actions[m + 2]  # Actions: [no-op, stop, move_north, move_south, move_east, move_west]

# Optional: Pathing Grid (8 dimensions if obs_pathing_grid=True)
if self.obs_pathing_grid:
    move_feats[ind : ind + self.n_obs_pathing] = self.get_surrounding_pathing(unit)
    # Returns pathing values of 8 surrounding points

# Optional: Terrain Height (9 dimensions if obs_terrain_height=True)  
if self.obs_terrain_height:
    move_feats[ind:] = self.get_surrounding_height(unit)
    # Returns height values of 9 surrounding points (8 + self)
```

**Data Format:** Binary (availability) for movement actions, float [0-1] for pathing/terrain

---

### 2.2 **Enemy Features** (enemy_feats)
```python
enemy_feats_dim = self.get_obs_enemy_feats_size()  # (n_enemies, n_features)
enemy_feats = np.zeros((self.n_enemies, enemy_feats_dim), dtype=np.float32)

# For each enemy unit
for e_id, e_unit in self.enemies.items():
    e_x = e_unit.pos.x
    e_y = e_unit.pos.y
    dist = self.distance(x, y, e_x, e_y)
    
    # Check visibility (based on sight range or conic FOV)
    enemy_visible = (
        self.is_position_in_cone(agent_id, e_unit.pos) if self.conic_fov
        else dist < sight_range
    )
    
    if (enemy_visible and e_unit.health > 0) or (e_unit.health > 0 and fully_observable):
        enemy_feats[e_id, 0] = avail_actions[self.n_actions_no_attack + e_id]  # shootable
        enemy_feats[e_id, 1] = dist / sight_range                             # distance (normalized)
        enemy_feats[e_id, 2] = (e_x - x) / sight_range                        # relative X
        enemy_feats[e_id, 3] = (e_y - y) / sight_range                        # relative Y
        
        ind = 4
        
        # Optional: Health
        if self.obs_all_health:
            enemy_feats[e_id, ind] = e_unit.health / e_unit.health_max
            ind += 1
        
        # Optional: Shield (for Protoss)
        if self.shield_bits_enemy > 0:
            max_shield = self.unit_max_shield(e_unit)
            enemy_feats[e_id, ind] = e_unit.shield / max_shield
            ind += 1
        
        # Optional: Unit Type (one-hot encoded)
        if self.unit_type_bits > 0:
            type_id = self.get_unit_type_id(e_unit, False)
            enemy_feats[e_id, ind + type_id] = 1  # One-hot
```

**Enemy Features Order:**
1. `shootable` - Binary, if attack is available
2. `distance` - Float [0-1], normalized by sight_range
3. `relative_x` - Float [-1 to 1], normalized by sight_range
4. `relative_y` - Float [-1 to 1], normalized by sight_range
5. `health` - Float [0-1], health/health_max (if obs_all_health=True)
6. `shield` - Float [0-1], shield/max_shield (if shield_bits_enemy > 0)
7. `unit_type` - One-hot encoding (if unit_type_bits > 0)

---

### 2.3 **Ally Features** (ally_feats)
```python
ally_feats_dim = self.get_obs_ally_feats_size()  # ((n_agents - 1), n_features)
ally_feats = np.zeros((self.n_agents - 1, ally_feats_dim), dtype=np.float32)

# For each other agent (excluding self)
al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]
for i, al_id in enumerate(al_ids):
    al_unit = self.get_unit_by_id(al_id)
    al_x = al_unit.pos.x
    al_y = al_unit.pos.y
    dist = self.distance(x, y, al_x, al_y)
    
    # Check visibility
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
        
        # Optional: Health
        if self.obs_all_health:
            if not self.stochastic_health:
                ally_feats[i, ind] = al_unit.health / al_unit.health_max
            elif self.observe_teammate_health:
                ally_feats[i, ind] = self._compute_health(al_id, al_unit)
            ind += 1
        
        # Optional: Shield
        if self.shield_bits_ally > 0:
            max_shield = self.unit_max_shield(al_unit)
            ally_feats[i, ind] = al_unit.shield / max_shield
            ind += 1
        
        # Optional: Attack Probability (stochastic attack capability)
        if self.stochastic_attack and self.observe_attack_probs:
            ally_feats[i, ind] = self.agent_attack_probabilities[al_id]
            ind += 1
        
        # Optional: Health Level (stochastic health capability)
        if self.stochastic_health and self.observe_teammate_health:
            ally_feats[i, ind] = self.agent_health_levels[al_id]
            ind += 1
        
        # Optional: Unit Type (one-hot)
        if self.unit_type_bits > 0:
            type_id = self.get_unit_type_id(al_unit, True)
            ally_feats[i, ind + type_id] = 1
            ind += self.unit_type_bits
        
        # Optional: Last Action (one-hot)
        if self.obs_last_action:
            ally_feats[i, ind:] = self.last_action[al_id]
```

**Ally Features Order:**
1. `visible` - Binary, visibility status
2. `distance` - Float [0-1], normalized by sight_range
3. `relative_x` - Float [-1 to 1], normalized by sight_range
4. `relative_y` - Float [-1 to 1], normalized by sight_range
5. `health` - Float [0-1] (if obs_all_health=True)
6. `shield` - Float [0-1] (if shield_bits_ally > 0)
7. `attack_probability` - Float [0-1] (if stochastic_attack=True and observe_attack_probs=True)
8. `health_level` - Float [0-1] (if stochastic_health=True and observe_teammate_health=True)
9. `unit_type` - One-hot encoding (if unit_type_bits > 0)
10. `last_action` - One-hot encoding of n_actions (if obs_last_action=True)

---

### 2.4 **Own Features** (own_feats)
```python
own_feats_dim = self.get_obs_own_feats_size()
own_feats = np.zeros(own_feats_dim, dtype=np.float32)

ind = 0

# Optional: Own Health
if self.obs_own_health:
    if not self.stochastic_health:
        own_feats[ind] = unit.health / unit.health_max
    else:
        own_feats[ind] = self._compute_health(agent_id, unit)
    ind += 1

# Optional: Own Shield
if self.shield_bits_ally > 0:
    max_shield = self.unit_max_shield(unit)
    own_feats[ind] = unit.shield / max_shield
    ind += 1

# Optional: Own Attack Probability
if self.stochastic_attack:
    own_feats[ind] = self.agent_attack_probabilities[agent_id]
    ind += 1

# Optional: Own Health Level
if self.stochastic_health:
    own_feats[ind] = self.agent_health_levels[agent_id]
    ind += 1

# Optional: Own Position
if self.obs_own_pos:
    own_feats[ind] = x / self.map_x
    own_feats[ind + 1] = y / self.map_y
    ind += 2

# Optional: Field of View Direction (conic FOV)
if self.conic_fov:
    own_feats[ind : ind + 2] = self.fov_directions[agent_id]
    ind += 2

# Optional: Unit Type (one-hot)
if self.unit_type_bits > 0:
    type_id = self.get_unit_type_id(unit, True)
    own_feats[ind + type_id] = 1
```

**Own Features Order:**
1. `health` - Float [0-1] (if obs_own_health=True, default=True)
2. `shield` - Float [0-1] (if shield_bits_ally > 0)
3. `attack_probability` - Float [0-1] (if stochastic_attack=True)
4. `health_level` - Float [0-1] (if stochastic_health=True)
5. `position_x` - Float [0-1], normalized by map width (if obs_own_pos=True)
6. `position_y` - Float [0-1], normalized by map height (if obs_own_pos=True)
7. `fov_x` - Float (if conic_fov=True)
8. `fov_y` - Float (if conic_fov=True)
9. `unit_type` - One-hot encoding (if unit_type_bits > 0)

---

### 2.5 **Optional: Timestep** 
```python
if self.obs_timestep_number:
    agent_obs = np.append(agent_obs, self._episode_steps / self.episode_limit)
```

**Format:** Float [0-1], normalized timestep

---

## 3. Final Observation Construction

```python
# Concatenate all components in order
agent_obs = np.concatenate((
    move_feats.flatten(),
    enemy_feats.flatten(),
    ally_feats.flatten(),
    own_feats.flatten(),
))

# Optional: Add timestep
if self.obs_timestep_number:
    agent_obs = np.append(agent_obs, self._episode_steps / self.episode_limit)

return agent_obs  # dtype=np.float32
```

**Output Type:** `numpy.ndarray` with dtype `float32`

---

## 4. Observation Size Calculation

```python
def get_obs_size(self):
    """Returns the size of the observation."""
    own_feats = self.get_obs_own_feats_size()
    move_feats = self.get_obs_move_feats_size()
    n_enemies, n_enemy_feats = self.get_obs_enemy_feats_size()
    n_allies, n_ally_feats = self.get_obs_ally_feats_size()
    
    enemy_feats = n_enemies * n_enemy_feats
    ally_feats = n_allies * n_ally_feats
    all_feats = move_feats + enemy_feats + ally_feats + own_feats
    
    timestep_feats = 1 if self.obs_timestep_number else 0
    all_feats += timestep_feats
    
    return [all_feats, [n_allies, n_ally_feats], [n_enemies, n_enemy_feats], [1, move_feats], [1, own_feats+timestep_feats]]
```

---

## 5. Default Configuration Example (8m map)

For the standard "8m" scenario (8 Marines vs 8 Marines):

```python
StarCraft2Env(
    map_name="8m",
    obs_all_health=True,          # Include all unit health
    obs_own_health=True,          # Include own health
    obs_last_action=False,        # Don't include last action
    obs_pathing_grid=False,       # Don't include pathing
    obs_terrain_height=False,     # Don't include terrain
    obs_timestep_number=False,    # Don't include timestep
    obs_own_pos=False,            # Don't include position
)
```

**Resulting Observation Size (8m scenario):**
- Movement features: 4 (4 move actions)
- Enemy features: 8 enemies × 5 features = 40 (shootable, distance, rel_x, rel_y, health)
- Ally features: 7 allies × 4 features = 28 (visible, distance, rel_x, rel_y)
- Own features: 1 (health)
- **Total: 73 elements**

---

## 6. Feature Normalization

| Feature | Range | Formula | Notes |
|---------|-------|---------|-------|
| Distance | [0-1] | `distance / sight_range` | Normalized by agent's sight range |
| Relative X | [-1, 1] | `(x_enemy - x_agent) / sight_range` | Can be outside [-1,1] if beyond sight range |
| Relative Y | [-1, 1] | `(y_enemy - y_agent) / sight_range` | Can be outside [-1,1] if beyond sight range |
| Health | [0-1] | `unit.health / unit.health_max` | Proportional health |
| Shield | [0-1] | `unit.shield / max_shield` | Proportional shield |
| Position X | [0-1] | `x / map_x` | Map-relative (used in state, not local obs) |
| Position Y | [0-1] | `y / map_y` | Map-relative (used in state, not local obs) |
| Timestep | [0-1] | `episode_steps / episode_limit` | Normalized episode progress |

---

## 7. Visibility Handling

### Sight Range Computation
```python
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
    return sight_range_map[unit.unit_type]
```

### Visibility Masks
- **Invisible units** are represented with **zero values** in the observation
- **Dead units** are treated as invisible (zeroed out)
- **Fully observable mode** (for debugging) ignores sight range

---

## 8. How Observations are Retrieved

### In `SMACv2_modified.py`:
```python
def step(self, actions):
    """A single environment step."""
    # ... execute actions ...
    
    # Get local observations for all agents
    local_obs = self.get_obs()  # Returns list of observations
    
    # Get global state
    global_state = np.array([self.env.get_state_agent(agent_id) 
                            for agent_id in range(self.env.n_agents)])
    
    # ... process rewards, dones, etc ...
    
    return local_obs, global_state, rewards, dones, infos, avail_actions

def get_obs(self):
    """Returns all agent observations in a list."""
    agents_obs = [self.get_obs_agent(i, fully_observable=self.fully_observable) 
                  for i in range(self.n_agents)]
    return agents_obs
```

---

## 9. Configuration Flags for Observation Design

| Flag | Default | Effect |
|------|---------|--------|
| `obs_all_health` | True | Include health of all visible units |
| `obs_own_health` | True | Include own agent's health (overridden if obs_all_health=True) |
| `obs_last_action` | False | Include one-hot encoding of last actions of allies |
| `obs_pathing_grid` | False | Include 8-element pathing grid around agent |
| `obs_terrain_height` | False | Include 9-element terrain height around agent |
| `obs_timestep_number` | False | Include normalized timestep |
| `obs_own_pos` | False | Include agent's absolute position (normalized) |
| `obs_instead_of_state` | False | Use concatenated observations as global state |
| `fully_observable` | False | Include information beyond sight range (debugging) |

---

## 10. Unit Type Encoding (One-Hot)

Unit types are encoded as one-hot vectors based on map type:

### MMM Map (Terran):
- Marine (index 0)
- Marauder (index 1)
- Medivac (index 2)

### Protoss Maps:
- Zealot (varies)
- Stalker (varies)
- Colossus (varies)

### Zerg Maps:
- Zergling (index 0)
- Hydralisk (index 1)
- Baneling (index 2)

**One-hot Encoding:** For unit_type_bits=3, a Marine is encoded as `[1, 0, 0]`

---

## Summary: Local Observation Structure

```
local_obs = np.concatenate([
    move_actions (4),           # Binary: North, South, East, West
    pathing_grid (0-8),        # Optional: surrounding pathing values
    terrain_height (0-9),      # Optional: surrounding terrain values
    ──────────────────────────
    enemy_0_features,          # shootable, distance, rel_x, rel_y, [health, shield, unit_type]
    enemy_1_features,
    ...
    enemy_n_features,
    ──────────────────────────
    ally_0_features,           # visible, distance, rel_x, rel_y, [health, shield, ...]
    ally_1_features,
    ...
    ally_n_features,
    ──────────────────────────
    own_features,              # [health, shield, attack_prob, health_level, pos_x, pos_y, fov_x, fov_y, unit_type]
    timestep (0-1),            # Optional
])
```

All values are **float32** and normalized to reasonable ranges (typically [0-1] or [-1, 1]).
