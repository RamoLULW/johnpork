from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple, Optional
from pydantic import BaseModel

Coord = Tuple[int, int]


class CellType(Enum):
    EMPTY = auto()
    PLANT = auto()
    ROCK = auto()
    WATER = auto()
    MUD = auto()
    ROAD = auto()
    STATION = auto()


class Weather(Enum):
    SUNNY = auto()
    CLOUDY = auto()
    RAIN = auto()


class PlantQuality(Enum):
    GOOD = 1
    BAD = 0


@dataclass
class Cell:
    x: int
    y: int
    cell_type: CellType
    has_plant: bool = False
    plant_quality: Optional[PlantQuality] = None


@dataclass
class TractorInternalState:
    id: int
    x: int
    y: int
    fuel: float
    max_fuel: float
    load: int
    capacity: int
    harvested_good: int = 0
    harvested_bad: int = 0
    total_fuel_used: float = 0.0


class HighLevelAction(Enum):
    GO_TO_PLANT = auto()
    GO_TO_STATION = auto()
    EXPLORE = auto()
    IDLE = auto()


class SimulationRequestDTO(BaseModel):
    N: int
    num_tractors: int = 2
    capacity: int = 10
    p_good: float = 0.7
    seed: int = 42
    max_ticks: int = 200
    use_qlearning: bool = True
    dynamic_weather: bool = True
    obstacle_intensity: float = 1.0
    fuel_capacity: float = 100.0


class PlantDTO(BaseModel):
    x: int
    y: int
    buena: bool
    mala: bool
    costo_mov: float
    transitable: bool


class TerrainDTO(BaseModel):
    x: int
    y: int
    cell_type: str
    costo_mov: float
    transitable: bool


class TractorStateDTO(BaseModel):
    id: int
    x: int
    y: int
    fuel: float
    load: int


class FrameDTO(BaseModel):
    tick: int
    tractors: List[TractorStateDTO]


class SimulationStatsDTO(BaseModel):
    total_plants: int
    good_harvested: int
    bad_harvested: int
    remaining_plants: int
    total_fuel_used: float
    ticks_run: int
    num_tractors: int
    grid_size: int


class SimulationResultDTO(BaseModel):
    initial_plants: List[PlantDTO]
    terrain: List[TerrainDTO]
    frames: List[FrameDTO]
    stats: SimulationStatsDTO
