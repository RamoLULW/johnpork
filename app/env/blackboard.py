from typing import Dict, Set, Optional
from app.core.types import Coord
from app.agents.interfaces import IBlackboard


class Blackboard(IBlackboard):
    def __init__(self):
        self.available_plants: Set[Coord] = set()
        self.claimed_plants: Dict[int, Coord] = {}
        self._tractors: Set[int] = set()

    def register_tractor(self, tractor_id: int) -> None:
        self._tractors.add(tractor_id)

    def add_plant(self, pos: Coord) -> None:
        if pos not in self.claimed_plants.values():
            self.available_plants.add(pos)

    def remove_plant(self, pos: Coord) -> None:
        self.available_plants.discard(pos)
        to_delete = []
        for tid, p in self.claimed_plants.items():
            if p == pos:
                to_delete.append(tid)
        for tid in to_delete:
            del self.claimed_plants[tid]

    def claim_plant(self, tractor_id: int, current_pos: Coord) -> Optional[Coord]:
        best = None
        best_dist = None
        for p in self.available_plants:
            d = abs(p[0] - current_pos[0]) + abs(p[1] - current_pos[1])
            if best is None or d < best_dist:
                best = p
                best_dist = d
        if best is None:
            return None
        self.available_plants.remove(best)
        self.claimed_plants[tractor_id] = best
        return best

    def release_plant(self, tractor_id: int, pos: Coord) -> None:
        if tractor_id in self.claimed_plants and self.claimed_plants[tractor_id] == pos:
            del self.claimed_plants[tractor_id]
            self.available_plants.add(pos)

    def remaining_plants(self) -> int:
        return len(self.available_plants) + len(self.claimed_plants)
