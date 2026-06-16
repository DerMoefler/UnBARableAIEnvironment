"""
Integrationstest OHNE Shared Memory:
- simuliert Shared-Memory-Daten komplett in Python
- simuliert einen 3v3 Pawn-Kampf
- nutzt die normale MAPPO-Trainingspipeline
- braucht KEIN shared_memory_reader_factory
- braucht KEINE echte Unit-IPC

Ziel:
- prüfen, ob Policy + ReplayBuffer + R_MAPPO + Trainingsloop funktionieren
- mit einer BAR-ähnlichen Unit-/Observation-Struktur
- ohne vom noch unfertigen Shared Memory abhängig zu sein

Start:
    PYTHONUNBUFFERED=1 uv run tests/test_bar_3v3_training_no_shm.py

Optional:
    PYTHONUNBUFFERED=1 uv run tests/test_bar_3v3_training_no_shm.py --num-episodes 10 --buffer-size 128 --debug-env
"""

from __future__ import annotations

import argparse
import inspect
import math
import traceback
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.environment.engine_session import BARUnitView, EngineSessionConfig
from src.train.policy import R_MAPPO_Policy
from src.train.replay_buffer import SharedReplayBuffer
from src.train.r_mappo import R_MAPPO


# -----------------------------------------------------------------------------
# Trainer-Args
# -----------------------------------------------------------------------------
class TrainerArgs:
    """Config namespace for R_MAPPO trainer."""

    def __init__(self) -> None:
        self.clip_param = 0.2
        self.ppo_epoch = 10
        self.num_mini_batch = 4
        self.data_chunk_length = 4
        self.value_loss_coef = 1.0
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.huber_delta = 10.0

        self.use_recurrent_policy = False
        self.use_naive_recurrent_policy = False
        self.use_max_grad_norm = True
        self.use_clipped_value_loss = True
        self.use_huber_loss = False
        self.use_popart = False
        self.use_valuenorm = False
        self.use_value_active_masks = True
        self.use_policy_active_masks = True


# -----------------------------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------------------------
def _prepare_reward(reward: Any, num_agents: int) -> np.ndarray:
    """
    Bringt Reward robust in Shape: (num_agents, 1)
    """
    if reward is None:
        return np.zeros((num_agents, 1), dtype=np.float32)

    arr = np.asarray(reward, dtype=np.float32)

    if arr.ndim == 0:
        arr = np.full((num_agents, 1), float(arr), dtype=np.float32)
    elif arr.ndim == 1:
        if arr.size == 1:
            arr = np.full((num_agents, 1), float(arr[0]), dtype=np.float32)
        elif arr.size == num_agents:
            arr = arr.reshape(num_agents, 1)
        else:
            fixed = np.zeros((num_agents,), dtype=np.float32)
            n = min(num_agents, arr.size)
            fixed[:n] = arr[:n]
            arr = fixed.reshape(num_agents, 1)
    else:
        arr = arr.reshape(num_agents, -1)
        arr = arr[:, :1]

    return arr.astype(np.float32)


def _prepare_dones(done_like: Any, num_agents: int) -> np.ndarray:
    """
    Bringt einzelne Agent-dones in Shape: (num_agents,)
    """
    if done_like is None:
        return np.zeros((num_agents,), dtype=bool)

    arr = np.asarray(done_like, dtype=bool)

    if arr.ndim == 0:
        return np.full((num_agents,), bool(arr), dtype=bool)

    arr = arr.reshape(-1)
    fixed = np.zeros((num_agents,), dtype=bool)
    n = min(num_agents, arr.size)
    fixed[:n] = arr[:n]
    return fixed


def _repeat_value(value_tensor: torch.Tensor, num_agents: int) -> np.ndarray:
    """
    Macht aus Critic-Output einen Buffer-kompatiblen Array.
    Zielshape: (num_agents, 1)
    """
    value_np = value_tensor.detach().cpu().numpy().reshape(-1)

    if value_np.size == 1:
        return np.full((num_agents, 1), float(value_np[0]), dtype=np.float32)

    fixed = np.zeros((num_agents,), dtype=np.float32)
    n = min(num_agents, value_np.size)
    fixed[:n] = value_np[:n]
    return fixed.reshape(num_agents, 1)


def _infer_buffer_insert_mode(buffer) -> str:
    """
    Erkennt grob, welche insert-Signatur der Buffer hat.
    """
    try:
        sig = inspect.signature(buffer.insert)
        params = list(sig.parameters.keys())

        if "rnn_states" in params or len(params) >= 10:
            return "extended"

        return "simple"
    except Exception:
        return "simple"


def _make_rnn_state_arrays(
    num_agents: int,
    recurrent_n: int = 1,
    hidden_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Erzeugt Default-RNN-Zustände für nicht/unklar rekurrente Policies.
    """
    rnn_shape = (num_agents, recurrent_n, hidden_size)
    return (
        np.zeros(rnn_shape, dtype=np.float32),
        np.zeros(rnn_shape, dtype=np.float32),
    )


def _extract_bad_masks(step_info: Any, num_agents: int) -> np.ndarray:
    """
    bad_masks:
    0.0 -> bad_transition=True
    1.0 -> normaler Übergang
    """
    bad_masks = np.ones((num_agents, 1), dtype=np.float32)

    if step_info is None:
        return bad_masks

    if isinstance(step_info, dict):
        if "bad_transition" in step_info:
            val = 0.0 if bool(step_info["bad_transition"]) else 1.0
            return np.full((num_agents, 1), val, dtype=np.float32)
        return bad_masks

    if isinstance(step_info, (list, tuple)):
        for i in range(min(num_agents, len(step_info))):
            item = step_info[i]
            if isinstance(item, dict):
                bad_masks[i, 0] = 0.0 if bool(item.get("bad_transition", False)) else 1.0
        return bad_masks

    return bad_masks


def _policy_sample_actions(
    policy,
    obs: np.ndarray,
    share_obs: np.ndarray,
    available_actions: np.ndarray,
    device: torch.device,
    num_agents: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robustes Aktionssampling:
    1) Wenn policy.get_actions(...) existiert, nutze es
    2) Sonst fallback auf actor/critic direkt
    """
    if hasattr(policy, "get_actions") and callable(policy.get_actions):
        rnn_states, rnn_states_critic = _make_rnn_state_arrays(num_agents)
        masks = np.ones((num_agents, 1), dtype=np.float32)

        value, action, action_log_prob, _, _ = policy.get_actions(
            np.repeat(share_obs[None, :], num_agents, axis=0),
            obs,
            rnn_states,
            rnn_states_critic,
            masks,
            available_actions,
        )

        value_np = value.detach().cpu().numpy().reshape(-1)
        action_np = action.detach().cpu().numpy().reshape(num_agents, 1)
        action_log_prob_np = action_log_prob.detach().cpu().numpy().reshape(num_agents, 1)

        if value_np.size == 1:
            values = np.full((num_agents, 1), float(value_np[0]), dtype=np.float32)
        else:
            fixed = np.zeros((num_agents,), dtype=np.float32)
            n = min(num_agents, value_np.size)
            fixed[:n] = value_np[:n]
            values = fixed.reshape(num_agents, 1)

        return values, action_np, action_log_prob_np

    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)

    with torch.no_grad():
        action_logits = policy.actor(obs_tensor)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        actions = action_dist.sample()
        action_log_probs = action_dist.log_prob(actions)

        share_obs_tensor = torch.as_tensor(
            share_obs, dtype=torch.float32, device=device
        ).unsqueeze(0)
        values = policy.critic(share_obs_tensor)

    value_preds = _repeat_value(values, num_agents)
    action_np = actions.detach().cpu().numpy().reshape(num_agents, 1)
    action_log_prob_np = action_log_probs.detach().cpu().numpy().reshape(num_agents, 1)

    return value_preds, action_np, action_log_prob_np


# -----------------------------------------------------------------------------
# Simulierte BAR-Unit-Welt ohne Shared Memory
# -----------------------------------------------------------------------------
@dataclass
class SimConfig:
    """
    Konfiguration der simulierten 3v3 Pawn-Welt.
    """
    num_agents: int = 3
    obs_dim: int = 32
    action_dim: int = 7
    arena_size: float = 100.0
    max_steps: int = 128
    move_step: float = 4.0
    attack_range: float = 12.0
    attack_damage: float = 18.0
    max_health: float = 100.0
    sight_radius: float = 40.0
    team_spacing: float = 25.0
    seed: int = 42


class SimulatedBAR3v3PawnEnv:
    """
    BAR-ähnliches Environment ohne Shared Memory.

    Es simuliert:
    - 3 Verbündete Pawns
    - 3 Gegner-Pawns
    - einfache Bewegung
    - einfache Nah-/Distanz-Attacks auf den nächsten Gegner
    - Health, Death, Reward, Done, Available Actions

    Actions:
        0 = stay
        1 = move_up    (+z)
        2 = move_down  (-z)
        3 = move_left  (-x)
        4 = move_right (+x)
        5 = attack nearest enemy
        6 = retreat from nearest enemy
    """

    def __init__(
        self,
        sim_cfg: SimConfig,
        engine_cfg: Optional[EngineSessionConfig] = None,
        debug_env: bool = False,
    ):
        self.cfg = sim_cfg
        self.engine_cfg = engine_cfg or EngineSessionConfig()
        self.debug_env = debug_env
        self.rng = np.random.default_rng(self.cfg.seed)

        self.frame = 0
        self.step_count = 0

        self.ally_team_id = 0
        self.enemy_team_id = 1
        self.ally_ally_team_id = 0
        self.enemy_ally_team_id = 1

        self.units: Dict[int, BARUnitView] = {}
        self.ally_agent_unit_ids: List[int] = []
        self.enemy_unit_ids: List[int] = []

        self.prev_stats: Optional[Tuple[int, int, float, float]] = None

    # -------------------------------------------------------------------------
    # Weltaufbau
    # -------------------------------------------------------------------------
    def _make_unit(
        self,
        unit_id: int,
        team_id: int,
        ally_team_id: int,
        x: float,
        z: float,
        name: str = "pawn",
    ) -> BARUnitView:
        return BARUnitView(
            unit_id=unit_id,
            unit_def_id=1,
            unit_def_name=name,
            human_name=name.capitalize(),
            team_id=team_id,
            ally_team_id=ally_team_id,
            health=float(self.cfg.max_health),
            max_health=float(self.cfg.max_health),
            pos_x=float(x),
            pos_y=0.0,
            pos_z=float(z),
            los_radius=float(self.cfg.sight_radius),
            air_los_radius=float(self.cfg.sight_radius),
            is_dead=False,
            being_built=False,
            build_progress=1.0,
            capture_progress=0.0,
            paralyze_damage=0.0,
        )

    def _spawn_world(self):
        self.units.clear()
        self.ally_agent_unit_ids.clear()
        self.enemy_unit_ids.clear()

        center = self.cfg.arena_size / 2.0
        spacing = self.cfg.team_spacing

        # Ally links
        ally_positions = [
            (center - spacing, center - 10.0),
            (center - spacing, center),
            (center - spacing, center + 10.0),
        ]

        # Enemy rechts
        enemy_positions = [
            (center + spacing, center - 10.0),
            (center + spacing, center),
            (center + spacing, center + 10.0),
        ]

        next_id = 1000

        for i, (x, z) in enumerate(ally_positions):
            uid = next_id + i
            unit = self._make_unit(
                unit_id=uid,
                team_id=self.ally_team_id,
                ally_team_id=self.ally_ally_team_id,
                x=x,
                z=z,
                name="pawn",
            )
            self.units[uid] = unit
            self.ally_agent_unit_ids.append(uid)

        next_id = 2000
        for i, (x, z) in enumerate(enemy_positions):
            uid = next_id + i
            unit = self._make_unit(
                unit_id=uid,
                team_id=self.enemy_team_id,
                ally_team_id=self.enemy_ally_team_id,
                x=x,
                z=z,
                name="pawn",
            )
            self.units[uid] = unit
            self.enemy_unit_ids.append(uid)

        self.prev_stats = self._team_stats()

    # -------------------------------------------------------------------------
    # Unit-Zugriff
    # -------------------------------------------------------------------------
    def _get_unit(self, unit_id: int) -> Optional[BARUnitView]:
        return self.units.get(unit_id)

    def _alive_units(self) -> List[BARUnitView]:
        return [u for u in self.units.values() if not u.is_dead]

    def _team_units(self, ally_team_id: int) -> List[BARUnitView]:
        return [u for u in self._alive_units() if u.ally_team_id == ally_team_id]

    def _distance(self, a: BARUnitView, b: BARUnitView) -> float:
        return math.sqrt(
            (a.pos_x - b.pos_x) ** 2 +
            (a.pos_y - b.pos_y) ** 2 +
            (a.pos_z - b.pos_z) ** 2
        )

    def _nearest_enemy(self, unit: BARUnitView) -> Optional[BARUnitView]:
        enemies = [u for u in self._alive_units() if u.ally_team_id != unit.ally_team_id]
        if not enemies:
            return None
        enemies.sort(key=lambda e: self._distance(unit, e))
        return enemies[0]

    def _units_in_sight(self, unit: BARUnitView) -> List[BARUnitView]:
        out = []
        for other in self._alive_units():
            if other.unit_id == unit.unit_id:
                continue
            if self._distance(unit, other) <= unit.los_radius:
                out.append(other)
        return out

    def _replace_unit(self, unit: BARUnitView):
        self.units[unit.unit_id] = unit

    def _clamp_pos(self, x: float, z: float) -> Tuple[float, float]:
        x = float(np.clip(x, 0.0, self.cfg.arena_size))
        z = float(np.clip(z, 0.0, self.cfg.arena_size))
        return x, z

    # -------------------------------------------------------------------------
    # Beobachtung
    # -------------------------------------------------------------------------
    def _health_pct(self, u: Optional[BARUnitView]) -> float:
        if u is None or u.max_health <= 0:
            return 0.0
        return float(u.health / max(1e-6, u.max_health))

    def _rel_pos(self, a: BARUnitView, b: BARUnitView) -> np.ndarray:
        return np.array([b.pos_x - a.pos_x, b.pos_y - a.pos_y, b.pos_z - a.pos_z], dtype=np.float32)

    def _build_agent_obs(self, agent_unit_id: int) -> np.ndarray:
        me = self._get_unit(agent_unit_id)

        own_feat_size = 7
        enemy_feat_size = 5
        max_enemies = 3
        ally_feat_size = 5
        max_allies = 2

        own_feats = np.zeros((own_feat_size,), dtype=np.float32)
        enemy_features = np.zeros((max_enemies, enemy_feat_size), dtype=np.float32)
        ally_features = np.zeros((max_allies, ally_feat_size), dtype=np.float32)

        if me is not None and not me.is_dead:
            own_feats[:7] = np.array(
                [
                    float(me.unit_def_id),
                    float(me.pos_x),
                    float(me.pos_y),
                    float(me.pos_z),
                    float(me.health),
                    float(self._health_pct(me)),
                    float(me.los_radius),
                ],
                dtype=np.float32,
            )

            seen = self._units_in_sight(me)
            enemies = [u for u in seen if u.ally_team_id != me.ally_team_id]
            allies = [u for u in seen if u.ally_team_id == me.ally_team_id]

            enemies = sorted(enemies, key=lambda u: self._distance(me, u))[:max_enemies]
            allies = sorted(allies, key=lambda u: self._distance(me, u))[:max_allies]

            for i, enemy in enumerate(enemies):
                rel = self._rel_pos(me, enemy)
                enemy_features[i, :5] = np.array(
                    [
                        float(enemy.unit_def_id),
                        float(rel[0]),
                        float(rel[1]),
                        float(rel[2]),
                        float(enemy.health),
                    ],
                    dtype=np.float32,
                )

            for i, ally in enumerate(allies):
                rel = self._rel_pos(me, ally)
                ally_features[i, :5] = np.array(
                    [
                        float(ally.unit_def_id),
                        float(rel[0]),
                        float(rel[1]),
                        float(rel[2]),
                        float(ally.health),
                    ],
                    dtype=np.float32,
                )

        obs = np.concatenate(
            [
                own_feats.flatten(),
                enemy_features.flatten(),
                ally_features.flatten(),
            ]
        ).astype(np.float32)

        # Sichere obs_dim erzwingen
        if obs.size < self.cfg.obs_dim:
            padded = np.zeros((self.cfg.obs_dim,), dtype=np.float32)
            padded[:obs.size] = obs
            obs = padded
        elif obs.size > self.cfg.obs_dim:
            obs = obs[:self.cfg.obs_dim]

        return obs.astype(np.float32)

    def _build_obs(self) -> np.ndarray:
        obs = np.stack(
            [self._build_agent_obs(uid) for uid in self.ally_agent_unit_ids],
            axis=0,
        )
        return obs.astype(np.float32)

    def _build_share_obs(self, obs: np.ndarray) -> np.ndarray:
        return obs.mean(axis=0).astype(np.float32)

    def _build_available_actions(self) -> np.ndarray:
        # aktuell immer alle Aktionen erlaubt
        avail = np.ones((self.cfg.num_agents, self.cfg.action_dim), dtype=np.float32)
        return avail

    # -------------------------------------------------------------------------
    # Reward / Statistiken
    # -------------------------------------------------------------------------
    def _team_stats(self) -> Tuple[int, int, float, float]:
        ally_units = self._team_units(self.ally_ally_team_id)
        enemy_units = self._team_units(self.enemy_ally_team_id)

        ally_alive = len(ally_units)
        enemy_alive = len(enemy_units)
        ally_hp = float(sum(u.health for u in ally_units))
        enemy_hp = float(sum(u.health for u in enemy_units))

        return ally_alive, enemy_alive, ally_hp, enemy_hp

    # -------------------------------------------------------------------------
    # Simulationsschritt
    # -------------------------------------------------------------------------
    def _move_unit(self, unit: BARUnitView, dx: float, dz: float):
        if unit.is_dead:
            return
        new_x, new_z = self._clamp_pos(unit.pos_x + dx, unit.pos_z + dz)
        self._replace_unit(
            replace(unit, pos_x=new_x, pos_z=new_z)
        )

    def _damage_unit(self, unit: BARUnitView, damage: float):
        if unit.is_dead:
            return
        new_hp = max(0.0, unit.health - damage)
        new_dead = new_hp <= 0.0
        self._replace_unit(
            replace(unit, health=new_hp, is_dead=new_dead)
        )

    def _apply_agent_action(self, agent_unit_id: int, action: int):
        unit = self._get_unit(agent_unit_id)
        if unit is None or unit.is_dead:
            return

        enemy = self._nearest_enemy(unit)

        # 0 = stay
        if action == 0:
            return

        # 1 = move_up (+z)
        if action == 1:
            self._move_unit(unit, 0.0, self.cfg.move_step)
            return

        # 2 = move_down (-z)
        if action == 2:
            self._move_unit(unit, 0.0, -self.cfg.move_step)
            return

        # 3 = move_left (-x)
        if action == 3:
            self._move_unit(unit, -self.cfg.move_step, 0.0)
            return

        # 4 = move_right (+x)
        if action == 4:
            self._move_unit(unit, self.cfg.move_step, 0.0)
            return

        # 5 = attack nearest enemy
        if action == 5:
            if enemy is None or enemy.is_dead:
                return

            dist = self._distance(unit, enemy)
            if dist <= self.cfg.attack_range:
                self._damage_unit(enemy, self.cfg.attack_damage)
            else:
                # wenn außerhalb Range: ein Stück auf Gegner zu
                dx = enemy.pos_x - unit.pos_x
                dz = enemy.pos_z - unit.pos_z
                norm = max(1e-6, math.sqrt(dx * dx + dz * dz))
                self._move_unit(
                    unit,
                    self.cfg.move_step * dx / norm,
                    self.cfg.move_step * dz / norm,
                )
            return

        # 6 = retreat from nearest enemy
        if action == 6:
            if enemy is None or enemy.is_dead:
                return

            dx = unit.pos_x - enemy.pos_x
            dz = unit.pos_z - enemy.pos_z
            norm = max(1e-6, math.sqrt(dx * dx + dz * dz))
            self._move_unit(
                unit,
                self.cfg.move_step * dx / norm,
                self.cfg.move_step * dz / norm,
            )
            return

    def _enemy_policy_step(self):
        """
        Sehr einfache Gegnerlogik:
        - greife nächsten Ally an wenn in Range
        - sonst laufe auf nächsten Ally zu
        """
        for enemy_id in self.enemy_unit_ids:
            enemy = self._get_unit(enemy_id)
            if enemy is None or enemy.is_dead:
                continue

            allies = [u for u in self._alive_units() if u.ally_team_id != enemy.ally_team_id]
            if not allies:
                continue

            allies.sort(key=lambda a: self._distance(enemy, a))
            target = allies[0]
            dist = self._distance(enemy, target)

            if dist <= self.cfg.attack_range:
                self._damage_unit(target, self.cfg.attack_damage * 0.85)
            else:
                dx = target.pos_x - enemy.pos_x
                dz = target.pos_z - enemy.pos_z
                norm = max(1e-6, math.sqrt(dx * dx + dz * dz))
                self._move_unit(
                    enemy,
                    self.cfg.move_step * 0.9 * dx / norm,
                    self.cfg.move_step * 0.9 * dz / norm,
                )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def reset(self):
        self.frame = 0
        self.step_count = 0
        self._spawn_world()

        obs = self._build_obs()
        share_obs = self._build_share_obs(obs)
        available_actions = self._build_available_actions()

        info = {
            "mode": "simulated_no_shared_memory",
            "ally_team_id": self.ally_team_id,
            "enemy_team_id": self.enemy_team_id,
            "ally_agent_unit_ids": list(self.ally_agent_unit_ids),
            "enemy_unit_ids": list(self.enemy_unit_ids),
            "engine_cfg_startscript": str(self.engine_cfg.startscript),
            "engine_cfg_engine_exe": str(self.engine_cfg.engine_exe),
            "shared_memory": False,
        }

        if self.debug_env:
            print("DEBUG ENV RESET:", info, flush=True)

        # kompatibel zu deinem train/test-code:
        return obs, share_obs, info, available_actions

    def step(self, action):
        self.step_count += 1
        self.frame += 30

        actions = np.asarray(action).reshape(-1)
        if actions.size != self.cfg.num_agents:
            fixed = np.zeros((self.cfg.num_agents,), dtype=np.int64)
            n = min(self.cfg.num_agents, actions.size)
            fixed[:n] = actions[:n]
            actions = fixed

        prev_ally_alive, prev_enemy_alive, prev_ally_hp, prev_enemy_hp = self.prev_stats

        # 1) Ally-Agenten-Aktionen
        for i, unit_id in enumerate(self.ally_agent_unit_ids):
            self._apply_agent_action(unit_id, int(actions[i]))

        # 2) Gegner-KI
        self._enemy_policy_step()

        # 3) Neue Stats
        ally_alive, enemy_alive, ally_hp, enemy_hp = self._team_stats()
        self.prev_stats = (ally_alive, enemy_alive, ally_hp, enemy_hp)

        # Reward:
        # + für enemy damage/deaths
        # - für own damage/deaths
        reward_scalar = 0.0
        reward_scalar += (prev_enemy_alive - enemy_alive) * 2.0
        reward_scalar -= (prev_ally_alive - ally_alive) * 2.0
        reward_scalar += (prev_enemy_hp - enemy_hp) * 0.05
        reward_scalar -= (prev_ally_hp - ally_hp) * 0.05

        rewards = np.full((self.cfg.num_agents,), reward_scalar, dtype=np.float32)

        ally_dead = ally_alive == 0
        enemy_dead = enemy_alive == 0
        max_steps_reached = self.step_count >= self.cfg.max_steps

        terminated = np.full((self.cfg.num_agents,), ally_dead or enemy_dead, dtype=bool)
        truncated = np.full((self.cfg.num_agents,), max_steps_reached and not (ally_dead or enemy_dead), dtype=bool)

        obs = self._build_obs()
        share_obs = self._build_share_obs(obs)
        available_actions = self._build_available_actions()

        infos = []
        won = enemy_dead and not ally_dead
        for i in range(self.cfg.num_agents):
            unit = self._get_unit(self.ally_agent_unit_ids[i])
            infos.append(
                {
                    "bad_transition": bool(truncated[i]),
                    "won": bool(won),
                    "ally_alive": int(ally_alive),
                    "enemy_alive": int(enemy_alive),
                    "ally_hp_sum": float(ally_hp),
                    "enemy_hp_sum": float(enemy_hp),
                    "frame": int(self.frame),
                    "agent_unit_id": int(self.ally_agent_unit_ids[i]),
                    "agent_dead": bool(unit.is_dead if unit is not None else True),
                }
            )

        if self.debug_env:
            print(
                f"DEBUG ENV STEP={self.step_count} frame={self.frame} "
                f"ally_alive={ally_alive} enemy_alive={enemy_alive} "
                f"ally_hp={ally_hp:.2f} enemy_hp={enemy_hp:.2f} reward={reward_scalar:.4f}",
                flush=True,
            )

        return obs, share_obs, rewards, terminated, truncated, infos, available_actions

    def close(self):
        # keine echten Ressourcen offen
        pass


# -----------------------------------------------------------------------------
# Hauptprogramm
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="BAR 3v3 Pawn test WITHOUT shared memory (simulated unit snapshot)"
    )
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--num-mini-batch", type=int, default=4, help="Number of mini-batches")
    parser.add_argument("--buffer-size", type=int, default=128, help="Replay buffer size / max steps")
    parser.add_argument("--num-agents", type=int, default=3, help="Number of agents (3 for 3v3)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--obs-dim", type=int, default=32, help="Observation dimension")
    parser.add_argument("--action-dim", type=int, default=7, help="Action dimension")
    parser.add_argument("--max-steps", type=int, default=128, help="Max simulation steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug-env", action="store_true", help="Print simulated env debug info")
    parser.add_argument("--debug-shapes", action="store_true", help="Print tensor/array shapes")
    args = parser.parse_args()

    print(f"DEBUG: args = {args}", flush=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"DEBUG: device = {device}", flush=True)

    success = False
    env = None

    try:
        # rein kompatibel / projekt-nah importiert, aber OHNE Shared Memory Nutzung
        engine_cfg = EngineSessionConfig(
            startscript="startscripts/3PawnVs3Pawn.txt",
        )

        sim_cfg = SimConfig(
            num_agents=args.num_agents,
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            max_steps=min(args.max_steps, args.buffer_size),
            seed=args.seed,
        )

        print("Initializing simulated BAR 3v3 pawn environment (no shared memory)...", flush=True)
        env = SimulatedBAR3v3PawnEnv(
            sim_cfg=sim_cfg,
            engine_cfg=engine_cfg,
            debug_env=args.debug_env,
        )
        print("DEBUG: SimulatedBAR3v3PawnEnv created", flush=True)

        print("Initializing policy...", flush=True)
        policy = R_MAPPO_Policy(args.obs_dim, args.action_dim, device=device, lr=args.lr)
        print("DEBUG: policy created", flush=True)

        print("Initializing replay buffer...", flush=True)
        buffer = SharedReplayBuffer(
            num_agents=args.num_agents,
            obs_shape=(args.obs_dim,),
            action_shape=(1,),
            buffer_size=args.buffer_size,
            device=device,
        )
        print("DEBUG: replay buffer created", flush=True)

        insert_mode = _infer_buffer_insert_mode(buffer)
        print(f"DEBUG: buffer insert mode = {insert_mode}", flush=True)

        print("Initializing R_MAPPO trainer...", flush=True)
        trainer_args = TrainerArgs()
        trainer_args.num_mini_batch = args.num_mini_batch
        trainer = R_MAPPO(trainer_args, policy, device=device)
        print("DEBUG: trainer created", flush=True)

        print("\nStarting simulated 3v3 pawn training loop...", flush=True)

        recurrent_n = 1
        hidden_size = 1

        for episode in range(args.num_episodes):
            print(f"DEBUG: starting episode {episode + 1}", flush=True)

            reset_result = env.reset()
            print("DEBUG: env.reset() returned", flush=True)

            obs, share_obs, info, available_actions = reset_result
            print(f"DEBUG: initial env info = {info}", flush=True)

            episode_reward = 0.0
            done = False
            step_count = 0
            rnn_states, rnn_states_critic = _make_rnn_state_arrays(
                args.num_agents, recurrent_n=recurrent_n, hidden_size=hidden_size
            )

            trainer.prep_rollout()

            while not done and step_count < args.buffer_size:
                value_preds, actions, action_log_probs = _policy_sample_actions(
                    policy=policy,
                    obs=obs,
                    share_obs=share_obs,
                    available_actions=available_actions,
                    device=device,
                    num_agents=args.num_agents,
                )

                env_actions = actions.reshape(args.num_agents)
                step_result = env.step(env_actions)

                (
                    next_obs,
                    next_share_obs,
                    reward,
                    terminated,
                    truncated,
                    step_info,
                    next_available_actions,
                ) = step_result

                terminated_arr = _prepare_dones(terminated, args.num_agents)
                truncated_arr = _prepare_dones(truncated, args.num_agents)
                dones = np.logical_or(terminated_arr, truncated_arr)

                done_env = bool(np.all(dones))
                done = done_env

                reward_arr = _prepare_reward(reward, args.num_agents)

                masks = np.ones((args.num_agents, 1), dtype=np.float32)
                if done_env:
                    masks[:] = 0.0

                active_masks = np.ones((args.num_agents, 1), dtype=np.float32)
                active_masks[dones] = 0.0
                if done_env:
                    active_masks[:] = 1.0

                bad_masks = _extract_bad_masks(step_info, args.num_agents)

                if done_env:
                    rnn_states[:] = 0.0
                    rnn_states_critic[:] = 0.0

                if args.debug_shapes:
                    print(f"DEBUG obs shape = {obs.shape}", flush=True)
                    print(f"DEBUG share_obs shape = {share_obs.shape}", flush=True)
                    print(f"DEBUG actions shape = {actions.shape}", flush=True)
                    print(f"DEBUG reward_arr shape = {reward_arr.shape}", flush=True)
                    print(f"DEBUG masks shape = {masks.shape}", flush=True)
                    print(f"DEBUG active_masks shape = {active_masks.shape}", flush=True)
                    print(f"DEBUG bad_masks shape = {bad_masks.shape}", flush=True)

                if insert_mode == "extended":
                    try:
                        buffer.insert(
                            share_obs,
                            obs,
                            rnn_states,
                            rnn_states_critic,
                            actions,
                            action_log_probs,
                            value_preds,
                            reward_arr,
                            masks,
                            bad_masks,
                            active_masks,
                            available_actions,
                        )
                    except TypeError:
                        buffer.insert(
                            share_obs=share_obs,
                            obs=obs,
                            rnn_states=rnn_states,
                            rnn_states_critic=rnn_states_critic,
                            actions=actions,
                            action_log_probs=action_log_probs,
                            value_preds=value_preds,
                            rewards=reward_arr,
                            masks=masks,
                            bad_masks=bad_masks,
                            active_masks=active_masks,
                            available_actions=available_actions,
                        )
                else:
                    try:
                        buffer.insert(
                            share_obs=share_obs,
                            obs=obs,
                            actions=actions,
                            action_log_probs=action_log_probs,
                            value_preds=value_preds,
                            rewards=reward_arr,
                            masks=masks,
                            active_masks=active_masks,
                        )
                    except TypeError:
                        buffer.insert(
                            share_obs,
                            obs,
                            actions,
                            action_log_probs,
                            value_preds,
                            reward_arr,
                            masks,
                            active_masks,
                        )

                episode_reward += float(reward_arr.mean())
                step_count += 1

                obs = next_obs
                share_obs = next_share_obs
                available_actions = next_available_actions

                if isinstance(step_info, list) and len(step_info) > 0 and isinstance(step_info[0], dict):
                    if args.debug_env:
                        print(f"DEBUG ENV INFO[0]: {step_info[0]}", flush=True)

            with torch.no_grad():
                share_obs_tensor = torch.as_tensor(
                    share_obs, dtype=torch.float32, device=device
                ).unsqueeze(0)
                next_value_tensor = policy.critic(share_obs_tensor)

            next_value = _repeat_value(next_value_tensor, args.num_agents)

            try:
                buffer.compute_returns(next_value, gamma=args.gamma)
            except TypeError:
                buffer.compute_returns(next_value)

            print("DEBUG: buffer.compute_returns() done", flush=True)

            trainer.prep_training()
            print("DEBUG: trainer.prep_training() done", flush=True)

            train_info = trainer.train(buffer, update_actor=True)
            print("DEBUG: trainer.train() done", flush=True)

            buffer.reset()
            print("DEBUG: buffer.reset() done", flush=True)

            print(f"Episode {episode + 1}/{args.num_episodes}", flush=True)
            print(f"  Episode Reward: {episode_reward:.4f}", flush=True)
            print(f"  Steps: {step_count}", flush=True)
            print(f"  Value Loss: {train_info.get('value_loss', 0):.6f}", flush=True)
            print(f"  Policy Loss: {train_info.get('policy_loss', 0):.6f}", flush=True)
            print(f"  Entropy: {train_info.get('dist_entropy', 0):.6f}", flush=True)
            print("", flush=True)

        success = True

    except Exception as e:
        print("\nERROR: Exception occurred during simulated 3v3 pawn training!", flush=True)
        print(f"ERROR TYPE: {type(e).__name__}", flush=True)
        print(f"ERROR MSG : {e}", flush=True)
        print("\nTRACEBACK:", flush=True)
        traceback.print_exc()
        raise

    finally:
        if env is not None:
            try:
                env.close()
                print("DEBUG: env.close() done", flush=True)
            except Exception as close_err:
                print(f"WARNING: env.close() failed: {close_err}", flush=True)

        if success:
            print("Simulated 3v3 pawn training complete!", flush=True)
        else:
            print("Simulated 3v3 pawn training aborted due to an error.", flush=True)


if __name__ == "__main__":
    main()