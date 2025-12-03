import random
from typing import List, Tuple, Optional
from app.core.types import Cell, CellType, Weather, PlantQuality, Coord
from app.agents.interfaces import IEnvironment


class Environment(IEnvironment):
    def __init__(self, N: int, p_good: float, seed: int, obstacle_intensity: float = 1.0, dynamic_weather: bool = True, num_stations: int = 1):
        self.N = N
        self.rng = random.Random(seed)
        self.grid: List[List[Cell]] = []
        self.weather = Weather.SUNNY
        self.dynamic_weather = dynamic_weather
        self.obstacle_intensity = max(0.0, obstacle_intensity)
        self._stations: List[Coord] = []
        self._init_grid(p_good, num_stations)

    def _init_grid(self, p_good: float, num_stations: int) -> None:
        base_empty = 0.5
        base_plant = 0.3
        base_water = 0.1 * self.obstacle_intensity if self.dynamic_weather else 0.0
        base_rock = 0.1 * self.obstacle_intensity
        total = base_empty + base_plant + base_water + base_rock
        p_empty = base_empty / total
        p_plant = p_empty + base_plant / total
        p_water = p_plant + base_water / total
        for x in range(self.N):
            row: List[Cell] = []
            for y in range(self.N):
                r = self.rng.random()
                if r < p_empty:
                    cell_type = CellType.EMPTY
                    has_plant = False
                    plant_quality = None
                elif r < p_plant:
                    cell_type = CellType.PLANT
                    has_plant = True
                    plant_quality = PlantQuality.GOOD if self.rng.random() < p_good else PlantQuality.BAD
                elif r < p_water:
                    cell_type = CellType.WATER
                    has_plant = False
                    plant_quality = None
                else:
                    cell_type = CellType.ROCK
                    has_plant = False
                    plant_quality = None
                row.append(Cell(x=x, y=y, cell_type=cell_type, has_plant=has_plant, plant_quality=plant_quality))
            self.grid.append(row)
        x = 0
        y = 0
        self.grid[x][y].cell_type = CellType.STATION
        self.grid[x][y].has_plant = False
        self.grid[x][y].plant_quality = None
        self._stations.append((x, y))
        for _ in range(self.N):
            rx = self.rng.randrange(self.N)
            for ry in range(self.N):
                cell = self.grid[rx][ry]
                if cell.cell_type not in (CellType.STATION, CellType.ROCK, CellType.WATER):
                    cell.cell_type = CellType.ROAD

    def get_size(self) -> int:
        return self.N

    def in_bounds(self, pos: Coord) -> bool:
        x, y = pos
        return 0 <= x < self.N and 0 <= y < self.N

    def is_transitable(self, pos: Coord) -> bool:
        if not self.in_bounds(pos):
            return False
        cell = self.grid[pos[0]][pos[1]]
        if cell.cell_type == CellType.ROCK:
            return False
        return True

    def movement_cost(self, pos: Coord) -> float:
        cell = self.grid[pos[0]][pos[1]]
        if cell.cell_type == CellType.ROAD:
            return 1.0
        if cell.cell_type == CellType.MUD:
            return 3.0
        if cell.cell_type == CellType.WATER:
            return 4.0
        if cell.cell_type == CellType.STATION:
            return 1.5
        return 2.0

    def get_neighbors(self, pos: Coord) -> List[Coord]:
        x, y = pos
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [p for p in candidates if self.in_bounds(p) and self.is_transitable(p)]

    def has_plant(self, pos: Coord) -> bool:
        cell = self.grid[pos[0]][pos[1]]
        return cell.has_plant

    def harvest_plant(self, pos: Coord) -> Optional[bool]:
        cell = self.grid[pos[0]][pos[1]]
        if not cell.has_plant:
            return None
        good = cell.plant_quality == PlantQuality.GOOD
        cell.has_plant = False
        cell.plant_quality = None
        return good

    def nearest_station(self, pos: Coord) -> Coord:
        best = None
        best_dist = None
        for s in self._stations:
            d = abs(s[0] - pos[0]) + abs(s[1] - pos[1])
            if best is None or d < best_dist:
                best = s
                best_dist = d
        return best

    def random_free_cell(self) -> Coord:
        while True:
            x = self.rng.randrange(self.N)
            y = self.rng.randrange(self.N)
            if self.is_transitable((x, y)):
                return x, y

    def update_climate(self) -> None:
        if not self.dynamic_weather:
            return
        r = self.rng.random()
        if r < 0.2:
            self.weather = Weather.RAIN
        elif r < 0.6:
            self.weather = Weather.CLOUDY
        else:
            self.weather = Weather.SUNNY
        if self.weather == Weather.RAIN:
            k = int(self.N * self.obstacle_intensity)
            k = max(1, k)
            for _ in range(k):
                x = self.rng.randrange(self.N)
                y = self.rng.randrange(self.N)
                cell = self.grid[x][y]
                if cell.cell_type in (CellType.EMPTY, CellType.PLANT, CellType.ROAD):
                    cell.cell_type = CellType.MUD
        else:
            for x in range(self.N):
                for y in range(self.N):
                    cell = self.grid[x][y]
                    if cell.cell_type == CellType.MUD:
                        cell.cell_type = CellType.EMPTY

    def iter_plants(self) -> List[Tuple[Coord, bool]]:
        result = []
        for x in range(self.N):
            for y in range(self.N):
                cell = self.grid[x][y]
                if cell.has_plant and cell.plant_quality is not None:
                    result.append(((x, y), cell.plant_quality == PlantQuality.GOOD))
        return result

    def iter_terrain(self) -> List[Tuple[Coord, str, float, bool]]:
        result = []
        for x in range(self.N):
            for y in range(self.N):
                pos = (x, y)
                cell = self.grid[x][y]
                result.append((pos, cell.cell_type.name, self.movement_cost(pos), self.is_transitable(pos)))
        return result
