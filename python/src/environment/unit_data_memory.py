from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

import logging
import bar_ai


@dataclass
class UnitMemoryEntry:
    unit: bar_ai.UnitData
    visible: bool
    last_seen_step: int
    confirmed_dead: bool = False


class UnitMemory:
    """
    Remembers copies of UnitData objects across LOS changes.

    If an enemy disappears from the current observation, its latest UnitData
    remains stored. Disappearance is interpreted as LOS loss, not death.
    """

    def __init__(self, training_team_id: int) -> None:
        self.training_team_id = int(training_team_id)
        self._enemies: Dict[int, UnitMemoryEntry] = {}
        self._current_step = 0

    def reset(
        self,
        visible_units: Iterable[bar_ai.UnitData],
        step: int = 0,
    ) -> None:
        """Reset memory at the start of an episode."""
        self._enemies.clear()
        self._current_step = int(step)
        self.update(visible_units, step)

    def update(
        self,
        visible_units: Iterable[bar_ai.UnitData],
        step: Optional[int] = None,
    ) -> None:
        """
        Update memory from the current LOS observation.

        Remembered enemies missing from the current observation are marked as
        invisible. Their last UnitData remains unchanged.
        """
        if step is None:
            self._current_step += 1
        else:
            self._current_step = int(step)

        visible_enemy_ids: Set[int] = set()

        for unit in visible_units:
            if int(unit.team_id) == self.training_team_id:
                continue

            unit_id = int(unit.unit_id)
            visible_enemy_ids.add(unit_id)

            previous = self._enemies.get(unit_id)

            confirmed_dead = bool(unit.is_dead)

            if previous is not None:
                confirmed_dead = (
                    confirmed_dead or previous.confirmed_dead
                )

            self._enemies[unit_id] = UnitMemoryEntry(
                unit=bar_ai.UnitData(unit),
                visible=True,
                last_seen_step=self._current_step,
                confirmed_dead=confirmed_dead,
            )

        # Nicht mehr sichtbare Units bleiben unverändert gespeichert.
        for unit_id, entry in self._enemies.items():
            if unit_id not in visible_enemy_ids:
                entry.visible = False

        logging.debug(
            "UnitMemory step=%d: visible=%d, hidden=%d, remembered=%d",
            self._current_step,
            self.visible_enemy_count,
            self.hidden_enemy_count,
            self.remembered_enemy_count,
        )

    def get_visible_entries(self) -> List[UnitMemoryEntry]:
        return [
            entry
            for entry in self._enemies.values()
            if entry.visible and not entry.confirmed_dead
        ]

    def get_hidden_entries(self) -> List[UnitMemoryEntry]:
        return [
            entry
            for entry in self._enemies.values()
            if not entry.visible and not entry.confirmed_dead
        ]

    def get_alive_units(self) -> List[bar_ai.UnitData]:
        """
        Return remembered enemies that have not been confirmed dead.

        This includes enemies outside LOS using their last known UnitData.
        """
        return [
            entry.unit
            for entry in self._enemies.values()
            if not entry.confirmed_dead
            and not bool(entry.unit.is_dead)
            and float(entry.unit.health) > 0.0
        ]

    @property
    def remembered_enemy_count(self) -> int:
        return len(self._enemies)

    @property
    def visible_enemy_count(self) -> int:
        return len(self.get_visible_entries())

    @property
    def hidden_enemy_count(self) -> int:
        return len(self.get_hidden_entries())