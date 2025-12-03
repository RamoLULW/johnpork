from abc import ABC, abstractmethod
from typing import List, Optional
from app.core.types import Coord, TractorInternalState, HighLevelAction


class IEnvironment(ABC):
    @abstractmethod
    def get_size(self) -> int:
        ...

    @abstractmethod
    def is_transitable(self, pos: Coord) -> bool:
        ...

    @abstractmethod
    def movement_cost(self, pos: Coord) -> float:
        ...

    @abstractmethod
    def get_neighbors(self, pos: Coord) -> List[Coord]:
        ...

    @abstractmethod
    def has_plant(self, pos: Coord) -> bool:
        ...

    @abstractmethod
    def harvest_plant(self, pos: Coord) -> Optional[bool]:
        ...

    @abstractmethod
    def nearest_station(self, pos: Coord) -> Coord:
        ...

    @abstractmethod
    def random_free_cell(self) -> Coord:
        ...

    @abstractmethod
    def update_climate(self) -> None:
        ...


class IBlackboard(ABC):
    @abstractmethod
    def register_tractor(self, tractor_id: int) -> None:
        ...

    @abstractmethod
    def add_plant(self, pos: Coord) -> None:
        ...

    @abstractmethod
    def remove_plant(self, pos: Coord) -> None:
        ...

    @abstractmethod
    def claim_plant(self, tractor_id: int, current_pos: Coord) -> Optional[Coord]:
        ...

    @abstractmethod
    def release_plant(self, tractor_id: int, pos: Coord) -> None:
        ...

    @abstractmethod
    def remaining_plants(self) -> int:
        ...


class IHighLevelPolicy(ABC):
    @abstractmethod
    def select_action(self, state_key: tuple, available_actions: List[HighLevelAction]) -> HighLevelAction:
        ...

    @abstractmethod
    def update(self, prev_state_key: tuple, action: HighLevelAction, reward: float, next_state_key: tuple) -> None:
        ...


class ITractorAgent(ABC):
    @abstractmethod
    def get_internal_state(self) -> TractorInternalState:
        ...

    @abstractmethod
    def step(self) -> None:
        ...
