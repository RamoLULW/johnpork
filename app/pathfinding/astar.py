from typing import Dict, List, Optional, Tuple
import heapq
from app.core.types import Coord
from app.agents.interfaces import IEnvironment


def reconstruct_path(came_from: Dict[Coord, Coord], current: Coord) -> List[Coord]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(env: IEnvironment, start: Coord, goal: Coord) -> Optional[List[Coord]]:
    if start == goal:
        return [start]
    open_set: List[Tuple[float, Coord]] = []
    heapq.heappush(open_set, (0.0, start))
    came_from: Dict[Coord, Coord] = {}
    g_score: Dict[Coord, float] = {start: 0.0}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            return reconstruct_path(came_from, current)
        for neighbor in env.get_neighbors(current):
            tentative_g = g_score[current] + env.movement_cost(neighbor)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                f = tentative_g + h
                heapq.heappush(open_set, (f, neighbor))
    return None
